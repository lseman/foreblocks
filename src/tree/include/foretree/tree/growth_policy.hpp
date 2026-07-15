#pragma once

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

#include "foretree/split/split_helpers.hpp"

namespace foretree {

// Stateless scheduling policies. Tree mutation and split evaluation stay in
// the trainer and are supplied as callbacks, while queue/level mechanics live
// here and can be tested independently.
class GrowthPolicy {
public:
    using Candidate = splitx::Candidate;

    template <class Evaluate, class Accept, class Apply, class Finalize, class Children, class Priority>
    static void leaf_wise(int root, int max_leaves, Evaluate&& evaluate, Accept&& accept, Apply&& apply,
                          Finalize&& finalize, Children&& children, Priority&& priority) {
        struct Item {
            double priority;
            int node;
            int sequence;
            Candidate split;
            bool operator<(const Item& other) const {
                if (priority != other.priority)
                    return priority > other.priority;
                return sequence > other.sequence;
            }
        };

        int sequence = 0;
        std::priority_queue<Item> queue;
        auto enqueue = [&](int node) {
            Candidate split;
            if (!evaluate(node, split) || !accept(node, split)) {
                finalize(node);
                return;
            }
            queue.push(Item{-priority(node, split), node, sequence++, std::move(split)});
        };

        enqueue(root);
        int leaves = 1;
        while (!queue.empty() && leaves < std::max(1, max_leaves)) {
            Item item = queue.top();
            queue.pop();
            if (!accept(item.node, item.split)) {
                finalize(item.node);
                continue;
            }
            apply(item.node, item.split);
            const auto [left, right] = children(item.node);
            ++leaves;
            if (left >= 0)
                enqueue(left);
            if (right >= 0)
                enqueue(right);
        }
    }

    template <class Evaluate, class Accept, class Apply, class Finalize, class Children>
    static void level_wise(int root, int max_depth, Evaluate&& evaluate, Accept&& accept, Apply&& apply,
                           Finalize&& finalize, Children&& children) {
        std::vector<int> queue{root};
        size_t head = 0;
        int depth = 0;
        while (head < queue.size()) {
            const size_t level_end = queue.size();
            while (head < level_end) {
                const int node = queue[head++];
                Candidate split;
                if (!evaluate(node, split) || !accept(node, split)) {
                    finalize(node);
                    continue;
                }
                apply(node, split);
                const auto [left, right] = children(node);
                if (left >= 0)
                    queue.push_back(left);
                if (right >= 0)
                    queue.push_back(right);
            }
            if (++depth >= max_depth)
                break;
        }
    }

    template <class Evaluate, class Accept, class Apply, class Finalize, class Children>
    static void oblivious(int root, int max_depth, Evaluate&& evaluate, Accept&& accept, Apply&& apply,
                          Finalize&& finalize, Children&& children) {
        std::vector<int> level{root};
        for (int depth = 0; !level.empty() && depth < max_depth; ++depth) {
            Candidate best;
            bool found = false;
            for (int node : level) {
                Candidate split;
                if (evaluate(node, split) && accept(node, split) && (!found || split.gain > best.gain)) {
                    best = std::move(split);
                    found = true;
                }
            }
            if (!found) {
                for (int node : level)
                    finalize(node);
                return;
            }

            std::vector<int> next;
            for (size_t index = 0; index < level.size(); ++index) {
                Candidate split = best;
                if (index > 0)
                    split.gain = 0.0;
                apply(level[index], split);
                const auto [left, right] = children(level[index]);
                if (left >= 0)
                    next.push_back(left);
                if (right >= 0)
                    next.push_back(right);
            }
            level = std::move(next);
        }
        for (int node : level)
            finalize(node);
    }
};

} // namespace foretree
