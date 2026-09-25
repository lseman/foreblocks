// Whole-tree GPU trainer: root split equals a brute-force double-precision
// reference, growth is deterministic, and the device row partition routes
// every row (missing codes included) to the leaf that tree traversal reaches.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "foretree/gpu/cuda_histogram.hpp"
#include "foretree/gpu/gpu_trainer.hpp"

namespace {

constexpr int kN = 20000, kP = 6;
constexpr uint8_t kMissing = 63;  // codes 0..62 finite, 63 missing

struct Data {
    std::vector<uint8_t> codes;  // row-major
    std::vector<double> y;
};

Data make_data() {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> code(0, kMissing - 1);
    std::bernoulli_distribution missing(0.1);
    Data d;
    d.codes.resize(static_cast<size_t>(kN) * kP);
    d.y.resize(kN);
    for (int i = 0; i < kN; ++i) {
        for (int f = 0; f < kP; ++f)
            d.codes[static_cast<size_t>(i) * kP + f] =
                (f == 2 && missing(rng)) ? kMissing : static_cast<uint8_t>(code(rng));
        const int c1 = d.codes[static_cast<size_t>(i) * kP + 1];
        const int c2 = d.codes[static_cast<size_t>(i) * kP + 2];
        d.y[i] = (c1 > 40 ? 2.0 : 0.0) + (c2 == kMissing ? 1.0 : c2 * 0.01) +
                 std::normal_distribution<double>(0, 0.1)(rng);
    }
    return d;
}

double leaf_obj(double G, double H, double lambda) {
    return 0.5 * G * G / (H + lambda);
}

// Brute-force best root split for squared error at margin 0 (g = -y, h = 1).
void reference_root(const Data& d, int& feature, int& bin, bool& missing_left, double& gain) {
    const double lambda = 1.0;
    gain = -1e300;
    for (int f = 0; f < kP; ++f) {
        std::vector<double> Gb(kMissing + 1, 0.0), Hb(kMissing + 1, 0.0);
        for (int i = 0; i < kN; ++i) {
            const int c = d.codes[static_cast<size_t>(i) * kP + f];
            Gb[c] += -d.y[i];
            Hb[c] += 1.0;
        }
        double Gt = 0, Ht = 0;
        for (int b = 0; b <= kMissing; ++b) {
            Gt += Gb[b];
            Ht += Hb[b];
        }
        const double parent = leaf_obj(Gt, Ht, lambda);
        for (int ml = 1; ml >= 0; --ml) {
            double GL = ml ? Gb[kMissing] : 0.0, HL = ml ? Hb[kMissing] : 0.0;
            for (int t = 0; t < kMissing; ++t) {
                GL += Gb[t];
                HL += Hb[t];
                if (Hb[t] == 0 || HL < 1 || Ht - HL < 1)
                    continue;
                const double g = leaf_obj(GL, HL, lambda) + leaf_obj(Gt - GL, Ht - HL, lambda) - parent;
                if (g > gain) {
                    gain = g;
                    feature = f;
                    bin = t;
                    missing_left = ml;
                }
            }
        }
    }
}

int leaf_of(const std::vector<foretree::cuda::GpuTreeNode>& nodes, const uint8_t* row) {
    int id = 0;
    while (!nodes[id].is_leaf) {
        const int c = row[nodes[id].feature];
        const bool left = c == kMissing ? nodes[id].missing_left : c <= nodes[id].threshold;
        id = left ? nodes[id].left : nodes[id].right;
    }
    return id;
}

}  // namespace

int main() {
    if (!foretree::cuda::is_available())
        return 0;
    const Data d = make_data();
    const auto dataset = foretree::QuantizedDataset::from_u8(kN, kP, d.codes, kMissing);
    using foretree::cuda::GpuObjective;
    using foretree::cuda::GpuTreeParams;
    using foretree::cuda::GpuTreeTrainer;

    // Root split vs brute force.
    {
        GpuTreeTrainer trainer(dataset, d.y, {}, GpuObjective::SquaredError, 0.0);
        GpuTreeParams params;
        params.max_leaves = 2;
        params.max_depth = 0;
        const auto nodes = trainer.grow(params);
        assert(nodes.size() == 3 && !nodes[0].is_leaf);
        int f = -1, bin = -1;
        bool ml = true;
        double gain = 0;
        reference_root(d, f, bin, ml, gain);
        assert(nodes[0].feature == f && nodes[0].threshold == bin && nodes[0].missing_left == ml);
        assert(std::abs(nodes[0].gain - gain) < 1e-4 * std::abs(gain));
        assert(nodes[1].count + nodes[2].count == kN);
        assert(std::abs(nodes[0].H - kN) < 1e-3);
    }

    // Deterministic growth; partition matches traversal; leaf limits hold.
    GpuTreeParams params;
    params.max_leaves = 31;
    params.max_depth = 8;
    params.min_samples_leaf = 20;
    GpuTreeTrainer a(dataset, d.y, {}, GpuObjective::SquaredError, 0.5);
    GpuTreeTrainer b(dataset, d.y, {}, GpuObjective::SquaredError, 0.5);
    for (int tree = 0; tree < 3; ++tree) {
        const auto na = a.grow(params);
        const auto nb = b.grow(params);
        assert(na.size() == nb.size());
        int leaves = 0;
        for (size_t i = 0; i < na.size(); ++i) {
            assert(na[i].feature == nb[i].feature && na[i].threshold == nb[i].threshold);
            assert(na[i].G == nb[i].G && na[i].count == nb[i].count);
            if (na[i].is_leaf) {
                ++leaves;
                assert(na[i].count >= params.min_samples_leaf);
            }
        }
        assert(leaves <= params.max_leaves);

        // values[node] = node id + 1: each row's margin gains exactly its leaf's id + 1.
        const auto before = a.margins();
        std::vector<double> values(na.size());
        for (size_t i = 0; i < na.size(); ++i)
            values[i] = static_cast<double>(i + 1);
        a.update_margins(na, values, 1.0);
        const auto after = a.margins();
        for (int i = 0; i < kN; ++i) {
            const int leaf = leaf_of(na, &d.codes[static_cast<size_t>(i) * kP]);
            assert(std::abs((after[i] - before[i]) - (leaf + 1)) < 1e-3);
        }
        b.update_margins(nb, values, 1.0);
    }
    return 0;
}
