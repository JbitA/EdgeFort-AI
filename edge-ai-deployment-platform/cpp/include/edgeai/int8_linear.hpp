#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>
namespace edgeai {
std::int32_t dot_i8(std::span<const std::int8_t> a, std::span<const std::int8_t> b);
std::vector<float> linear_i8(std::span<const std::int8_t> x,
                             std::span<const std::int8_t> weights,
                             std::span<const float> weight_scales,
                             std::span<const float> bias,
                             std::size_t outputs,
                             std::size_t inputs,
                             float input_scale);
}
