#include <cassert>
#include <map>
#include <utility>
#include <vector>

#include "foretree/tree/growth_policy.hpp"

int main() {
    std::map<int, std::pair<int, int>> children;
    std::vector<int> split_nodes;
    std::vector<int> finalized;
    int next = 1;

    auto evaluate = [](int node, foretree::splitx::Candidate& split) {
        if (node >= 3)
            return false;
        split.feat = 0;
        split.thr = 0;
        split.gain = 10.0 - node;
        return true;
    };
    auto accept = [](int, const auto&) { return true; };
    auto apply = [&](int node, const auto&) {
        split_nodes.push_back(node);
        children[node] = {next, next + 1};
        next += 2;
    };
    auto finalize = [&](int node) { finalized.push_back(node); };
    auto get_children = [&](int node) { return children.at(node); };
    auto priority = [](int, const auto& split) { return split.gain; };

    foretree::GrowthPolicy::leaf_wise(0, 3, evaluate, accept, apply, finalize, get_children, priority);
    assert(split_nodes == std::vector<int>({0, 1}));
    assert((children.at(0) == std::pair<int, int>(1, 2)));
    assert((children.at(1) == std::pair<int, int>(3, 4)));
}
