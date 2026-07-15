#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

namespace foretree {

enum class FeatureType : uint8_t { Numerical, Categorical, OrderedCategorical };
enum class QuantizedCodeWidth : uint8_t { UInt8 = 1, UInt16 = 2 };

struct DenseDatasetView {
    const double* values = nullptr;
    int rows = 0;
    int features = 0;

    void validate() const {
        if (!values)
            throw std::invalid_argument("DenseDatasetView: values is null");
        if (rows <= 0 || features <= 0)
            throw std::invalid_argument("DenseDatasetView: invalid shape");
        if (static_cast<size_t>(rows) > std::numeric_limits<size_t>::max() / static_cast<size_t>(features)) {
            throw std::overflow_error("DenseDatasetView: shape overflow");
        }
    }
};

class QuantizedDataset {
public:
    using UInt8Storage = std::vector<uint8_t>;
    using UInt16Storage = std::vector<uint16_t>;

    QuantizedDataset() = default;

    static QuantizedDataset from_u16(int rows, int features, UInt16Storage codes, uint16_t maximum_code,
                                     std::vector<uint16_t> missing_codes = {},
                                     std::vector<FeatureType> feature_types = {}) {
        validate_shape_(rows, features, codes.size());
        QuantizedDataset dataset;
        dataset.rows_ = rows;
        dataset.features_ = features;
        dataset.maximum_code_ = maximum_code;
        dataset.missing_codes_ =
            normalize_metadata_(std::move(missing_codes), features, static_cast<uint16_t>(maximum_code));
        dataset.feature_types_ = normalize_metadata_(std::move(feature_types), features, FeatureType::Numerical);

        if (maximum_code <= std::numeric_limits<uint8_t>::max()) {
            UInt8Storage compact(codes.size());
            std::transform(codes.begin(), codes.end(), compact.begin(),
                           [](uint16_t code) { return static_cast<uint8_t>(code); });
            dataset.storage_ = std::move(compact);
        } else {
            dataset.storage_ = std::move(codes);
        }
        return dataset;
    }

    static QuantizedDataset from_u8(int rows, int features, UInt8Storage codes, uint8_t maximum_code,
                                    std::vector<uint16_t> missing_codes = {},
                                    std::vector<FeatureType> feature_types = {}) {
        validate_shape_(rows, features, codes.size());
        QuantizedDataset dataset;
        dataset.rows_ = rows;
        dataset.features_ = features;
        dataset.maximum_code_ = maximum_code;
        dataset.missing_codes_ =
            normalize_metadata_(std::move(missing_codes), features, static_cast<uint16_t>(maximum_code));
        dataset.feature_types_ = normalize_metadata_(std::move(feature_types), features, FeatureType::Numerical);
        dataset.storage_ = std::move(codes);
        return dataset;
    }

    [[nodiscard]] int rows() const noexcept {
        return rows_;
    }
    [[nodiscard]] int features() const noexcept {
        return features_;
    }
    [[nodiscard]] uint16_t maximum_code() const noexcept {
        return maximum_code_;
    }
    [[nodiscard]] QuantizedCodeWidth code_width() const noexcept {
        return std::holds_alternative<UInt8Storage>(storage_) ? QuantizedCodeWidth::UInt8 : QuantizedCodeWidth::UInt16;
    }
    [[nodiscard]] size_t size() const noexcept {
        return std::visit([](const auto& values) { return values.size(); }, storage_);
    }
    [[nodiscard]] size_t bytes() const noexcept {
        return std::visit(
            [](const auto& values) {
                return values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type);
            },
            storage_);
    }

    [[nodiscard]] uint16_t code(int row, int feature) const noexcept {
        const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(features_) + static_cast<size_t>(feature);
        return std::visit([offset](const auto& values) { return static_cast<uint16_t>(values[offset]); }, storage_);
    }

    [[nodiscard]] uint16_t missing_code(int feature) const noexcept {
        return missing_codes_[static_cast<size_t>(feature)];
    }

    [[nodiscard]] FeatureType feature_type(int feature) const noexcept {
        return feature_types_[static_cast<size_t>(feature)];
    }

    [[nodiscard]] std::span<const FeatureType> feature_types() const noexcept {
        return feature_types_;
    }

    [[nodiscard]] std::span<const uint16_t> missing_codes() const noexcept {
        return missing_codes_;
    }

    template <class Function> decltype(auto) visit_codes(Function&& function) const {
        return std::visit(
            [&](const auto& values) -> decltype(auto) {
                return function(std::span<const typename std::decay_t<decltype(values)>::value_type>(values));
            },
            storage_);
    }

    // Lazily materialized feature-major codes. The row-major representation
    // remains canonical for prediction and partitioning, while this view gives
    // histogram kernels contiguous access to one feature at a time.
    template <class Function> decltype(auto) visit_feature_major_codes(Function&& function) const {
        ensure_feature_major_();
        return std::visit(
            [&](const auto& values) -> decltype(auto) {
                using Value = typename std::decay_t<decltype(values)>::value_type;
                return function(std::span<const Value>(values));
            },
            feature_major_cache_->storage);
    }

    [[nodiscard]] bool has_feature_major_cache() const noexcept {
        return feature_major_cache_ && feature_major_cache_->ready.load(std::memory_order_acquire);
    }

    [[nodiscard]] size_t feature_major_bytes() const noexcept {
        if (!has_feature_major_cache())
            return 0;
        return std::visit(
            [](const auto& values) {
                return values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type);
            },
            feature_major_cache_->storage);
    }

    [[nodiscard]] UInt16Storage to_u16() const {
        return std::visit([](const auto& values) { return UInt16Storage(values.begin(), values.end()); }, storage_);
    }

    void serialize(std::ostream& output) const {
        constexpr uint32_t magic = 0x46514431U; // FQD1
        write_(output, magic);
        write_(output, static_cast<uint32_t>(rows_));
        write_(output, static_cast<uint32_t>(features_));
        write_(output, static_cast<uint8_t>(code_width()));
        write_(output, maximum_code_);
        output.write(reinterpret_cast<const char*>(missing_codes_.data()),
                     static_cast<std::streamsize>(missing_codes_.size() * sizeof(uint16_t)));
        for (FeatureType type : feature_types_)
            write_(output, static_cast<uint8_t>(type));
        visit_codes([&](auto codes) {
            output.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size_bytes()));
        });
        if (!output)
            throw std::runtime_error("QuantizedDataset::serialize failed");
    }

    static QuantizedDataset deserialize(std::istream& input) {
        constexpr uint32_t expected_magic = 0x46514431U;
        const uint32_t magic = read_<uint32_t>(input);
        if (magic != expected_magic)
            throw std::invalid_argument("QuantizedDataset: invalid serialized header");
        const auto rows = static_cast<int>(read_<uint32_t>(input));
        const auto features = static_cast<int>(read_<uint32_t>(input));
        const auto width = static_cast<QuantizedCodeWidth>(read_<uint8_t>(input));
        const uint16_t maximum_code = read_<uint16_t>(input);
        if (rows <= 0 || features <= 0)
            throw std::invalid_argument("QuantizedDataset: invalid serialized shape");
        std::vector<uint16_t> missing_codes(static_cast<size_t>(features));
        input.read(reinterpret_cast<char*>(missing_codes.data()),
                   static_cast<std::streamsize>(missing_codes.size() * sizeof(uint16_t)));
        std::vector<FeatureType> feature_types(static_cast<size_t>(features));
        for (auto& type : feature_types)
            type = static_cast<FeatureType>(read_<uint8_t>(input));
        const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(features);
        if (width == QuantizedCodeWidth::UInt8) {
            UInt8Storage codes(count);
            input.read(reinterpret_cast<char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
            if (!input)
                throw std::runtime_error("QuantizedDataset::deserialize truncated uint8 data");
            return from_u8(rows, features, std::move(codes), static_cast<uint8_t>(maximum_code),
                           std::move(missing_codes), std::move(feature_types));
        }
        if (width == QuantizedCodeWidth::UInt16) {
            UInt16Storage codes(count);
            input.read(reinterpret_cast<char*>(codes.data()),
                       static_cast<std::streamsize>(codes.size() * sizeof(uint16_t)));
            if (!input)
                throw std::runtime_error("QuantizedDataset::deserialize truncated uint16 data");
            return from_u16(rows, features, std::move(codes), maximum_code, std::move(missing_codes),
                            std::move(feature_types));
        }
        throw std::invalid_argument("QuantizedDataset: unsupported serialized code width");
    }

private:
    struct FeatureMajorCache {
        std::mutex mutex;
        std::variant<UInt8Storage, UInt16Storage> storage = UInt8Storage{};
        std::atomic_bool ready = false;
    };

    void ensure_feature_major_() const {
        auto& cache = *feature_major_cache_;
        std::lock_guard lock(cache.mutex);
        if (cache.ready.load(std::memory_order_relaxed))
            return;
        std::visit(
            [&](const auto& row_major) {
                using Storage = std::decay_t<decltype(row_major)>;
                Storage feature_major(row_major.size());
                for (int feature = 0; feature < features_; ++feature) {
                    const size_t column = static_cast<size_t>(feature) * static_cast<size_t>(rows_);
                    for (int row = 0; row < rows_; ++row) {
                        feature_major[column + static_cast<size_t>(row)] =
                            row_major[static_cast<size_t>(row) * static_cast<size_t>(features_) +
                                      static_cast<size_t>(feature)];
                    }
                }
                cache.storage = std::move(feature_major);
            },
            storage_);
        cache.ready.store(true, std::memory_order_release);
    }

    template <class T> static void write_(std::ostream& output, T value) {
        output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <class T> static T read_(std::istream& input) {
        T value{};
        input.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!input)
            throw std::runtime_error("QuantizedDataset::deserialize truncated header");
        return value;
    }

    template <class T> static std::vector<T> normalize_metadata_(std::vector<T> values, int features, T fallback) {
        if (values.empty())
            values.assign(static_cast<size_t>(features), fallback);
        if (values.size() != static_cast<size_t>(features)) {
            throw std::invalid_argument("QuantizedDataset: feature metadata size mismatch");
        }
        return values;
    }

    static void validate_shape_(int rows, int features, size_t size) {
        if (rows <= 0 || features <= 0)
            throw std::invalid_argument("QuantizedDataset: invalid shape");
        const size_t expected = static_cast<size_t>(rows) * static_cast<size_t>(features);
        if (size != expected)
            throw std::invalid_argument("QuantizedDataset: code count does not match shape");
    }

    int rows_ = 0;
    int features_ = 0;
    uint16_t maximum_code_ = 0;
    std::variant<UInt8Storage, UInt16Storage> storage_ = UInt8Storage{};
    std::vector<uint16_t> missing_codes_;
    std::vector<FeatureType> feature_types_;
    mutable std::shared_ptr<FeatureMajorCache> feature_major_cache_ = std::make_shared<FeatureMajorCache>();
};

using QuantizedDatasetPtr = std::shared_ptr<QuantizedDataset>;

} // namespace foretree
