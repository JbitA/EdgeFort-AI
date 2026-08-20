#include "edgeai/int8_linear.hpp"

#include <cmath>
#include <limits>

namespace edgeai {
std::int32_t dot_i8(std::span<const std::int8_t> a, std::span<const std::int8_t> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot_i8 shape mismatch");
    }
    std::int64_t acc = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    }
    if (acc < std::numeric_limits<std::int32_t>::min() ||
        acc > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("int8 accumulator overflow");
    }
    return static_cast<std::int32_t>(acc);
}

std::vector<float> linear_i8(std::span<const std::int8_t> x,
                             std::span<const std::int8_t> weights,
                             std::span<const float> weight_scales,
                             std::span<const float> bias,
                             std::size_t outputs,
                             std::size_t inputs,
                             float input_scale) {
    if (!std::isfinite(input_scale) || input_scale <= 0) {
        throw std::invalid_argument("invalid input scale");
    }
    if (outputs != 0 && inputs > std::numeric_limits<std::size_t>::max() / outputs) {
        throw std::invalid_argument("linear_i8 dimensions overflow size_t");
    }
    const auto required_weights = outputs * inputs;
    if (x.size() != inputs || weights.size() != required_weights ||
        weight_scales.size() != outputs || bias.size() != outputs) {
        throw std::invalid_argument("linear_i8 shape mismatch");
    }

    std::vector<float> out(outputs);
    for (std::size_t output = 0; output < outputs; ++output) {
        const auto row = weights.subspan(output * inputs, inputs);
        const auto acc = dot_i8(x, row);
        const auto scale = input_scale * weight_scales[output];
        if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(bias[output])) {
            throw std::invalid_argument("invalid model parameters");
        }
        out[output] = static_cast<float>(acc) * scale + bias[output];
        if (!std::isfinite(out[output])) {
            throw std::overflow_error("non-finite output");
        }
    }
    return out;
}
}  // namespace edgeai
