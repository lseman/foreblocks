#include <cassert>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

#include "foretree/core/data_binner.hpp"
#include "foretree/core/dataset.hpp"

static void check_node_override(int capacity) {
    foretree::DataBinner binner(3);
    foretree::EdgeSet edges;
    edges.edges_per_feat = {{0, 1, 2}, {0, 1}, {}};
    for (int cut = 0; cut <= capacity; ++cut)
        edges.edges_per_feat[2].push_back(cut);
    binner.register_edges("hist", edges);

    binner.set_node_override("hist", 7, 0, {0, 1});
    binner.set_node_override("hist", 7, 1, {0, 1, 2, 3});
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> values = {nan, nan, nan, 2, 2, 2};
    const auto local = binner.prebin_compact(values.data(), 2, 3, "hist", 7);
    const auto width = capacity > 255 ? foretree::QuantizedCodeWidth::UInt16 : foretree::QuantizedCodeWidth::UInt8;
    assert(local->code_width() == width);
    assert(local->to_u16() == std::vector<uint16_t>({1, 3, static_cast<uint16_t>(capacity), 0, 2, 2}));
    for (int feature = 0; feature < 3; ++feature)
        assert(local->missing_code(feature) == local->code(0, feature));

    const auto [codes, maximum] = binner.prebin(values.data(), 2, 3, "hist", 7);
    assert(maximum == capacity && *codes == local->to_u16());
    std::vector<uint16_t> into(values.size());
    assert(binner.prebin_into(values.data(), 2, 3, "hist", into.data(), 7) == capacity);
    assert(into == *codes);

    const auto other = binner.prebin_compact(values.data(), 2, 3, "hist", 8);
    assert(other->missing_code(0) == 2 && other->missing_code(1) == 1);
    binner.register_edges("hist", edges);
    assert(binner.prebin_compact(values.data(), 2, 3, "hist", 7)->to_u16() == other->to_u16());
}

int main() {
    check_node_override(3);
    check_node_override(256);
    std::vector<uint16_t> small_codes = {0, 1, 2, 3, 4, 5};
    auto compact = foretree::QuantizedDataset::from_u16(2, 3, std::move(small_codes), 5, {5, 5, 5});
    assert(compact.code_width() == foretree::QuantizedCodeWidth::UInt8);
    assert(compact.bytes() == 6);
    assert(compact.code(1, 2) == 5);
    assert(compact.to_u16() == std::vector<uint16_t>({0, 1, 2, 3, 4, 5}));
    assert(!compact.has_feature_major_cache());
    compact.visit_feature_major_codes([](auto codes) {
        using Code = typename decltype(codes)::value_type;
        assert(codes.size() == 6);
        assert(codes[0] == static_cast<Code>(0));
        assert(codes[1] == static_cast<Code>(3));
        assert(codes[2] == static_cast<Code>(1));
        assert(codes[3] == static_cast<Code>(4));
        assert(codes[4] == static_cast<Code>(2));
        assert(codes[5] == static_cast<Code>(5));
    });
    assert(compact.has_feature_major_cache());
    assert(compact.feature_major_bytes() == compact.bytes());

    std::vector<uint16_t> wide_codes = {0, 256, 12, 511};
    auto wide = foretree::QuantizedDataset::from_u16(2, 2, std::move(wide_codes), 511, {256, 511});
    assert(wide.code_width() == foretree::QuantizedCodeWidth::UInt16);
    assert(wide.bytes() == 8);
    assert(wide.code(0, 1) == 256);
    assert(wide.missing_code(1) == 511);

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    wide.serialize(buffer);
    buffer.seekg(0);
    auto restored = foretree::QuantizedDataset::deserialize(buffer);
    assert(restored.code_width() == foretree::QuantizedCodeWidth::UInt16);
    assert(restored.rows() == wide.rows());
    assert(restored.features() == wide.features());
    assert(restored.to_u16() == wide.to_u16());
    assert(restored.missing_code(0) == wide.missing_code(0));

    foretree::DataBinner binner(2);
    foretree::EdgeSet edges;
    edges.edges_per_feat = {{0.0, 1.0, 2.0}, {0.0, 10.0}};
    edges.feature_types = {foretree::FeatureType::Numerical, foretree::FeatureType::Categorical};
    binner.register_edges("hist", std::move(edges));
    const std::vector<double> values = {
        0.25,
        5.0,
        1.75,
        std::numeric_limits<double>::quiet_NaN(),
    };
    auto compact_binned = binner.prebin_compact(values.data(), 2, 2, "hist");
    assert(compact_binned->code_width() == foretree::QuantizedCodeWidth::UInt8);
    assert(compact_binned->to_u16() == std::vector<uint16_t>({0, 0, 1, 1}));
    assert(compact_binned->missing_code(0) == 2);
    assert(compact_binned->missing_code(1) == 1);
    assert(compact_binned->feature_type(0) == foretree::FeatureType::Numerical);
    assert(compact_binned->feature_type(1) == foretree::FeatureType::Categorical);

    std::vector<uint16_t> into(4, 999);
    const int maximum_missing = binner.prebin_into(values.data(), 2, 2, "hist", into.data());
    assert(maximum_missing == 2);
    assert(into == compact_binned->to_u16());

    // Tiny, uneven steps can pass the uniform-spacing tolerance. The stored
    // cuts still define the bins, including for values exactly on a cut.
    foretree::DataBinner small_scale(1);
    foretree::EdgeSet small_edges;
    small_edges.edges_per_feat = {{0.0, 1e-12, 9e-12, 1e-11}};
    small_scale.register_edges("hist", std::move(small_edges));
    const std::vector<double> probes = {
        std::numeric_limits<double>::lowest(), std::nextafter(1e-12, 0.0),
        1e-12, std::nextafter(9e-12, 0.0), 9e-12,
        std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity()
    };
    const auto encoded = small_scale.prebin_compact(probes.data(), probes.size(), 1, "hist");
    assert(encoded->to_u16() == std::vector<uint16_t>({0, 0, 1, 1, 2, 2, 3}));
}
