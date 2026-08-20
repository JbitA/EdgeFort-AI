#include "edgeai/int8_linear.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#define CHECK(x)                                                                                  \
    do {                                                                                          \
        if (!(x)) {                                                                               \
            std::cerr << "CHECK failed: " #x " at " << __FILE__ << ':' << __LINE__ << '\n';      \
            return 1;                                                                             \
        }                                                                                         \
    } while (0)

int main() {
    using namespace edgeai;

    std::vector<std::int8_t> a{1, -2, 3};
    std::vector<std::int8_t> b{4, 5, -6};
    CHECK(dot_i8(a, b) == -24);

    bool threw = false;
    try {
        std::vector<std::int8_t> c{1};
        dot_i8(a, c);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // 133,145 * 127 * 127 exceeds INT32_MAX. The native kernel must fail
    // deterministically rather than wrap an accumulator.
    std::vector<std::int8_t> overflow_a(133145, 127);
    std::vector<std::int8_t> overflow_b(133145, 127);
    threw = false;
    try {
        dot_i8(overflow_a, overflow_b);
    } catch (const std::overflow_error&) {
        threw = true;
    }
    CHECK(threw);

    std::vector<std::int8_t> x{10, 20};
    std::vector<std::int8_t> w{2, 3, -4, 5};
    std::vector<float> scales{0.1F, 0.2F};
    std::vector<float> bias{1.F, -2.F};
    auto y = linear_i8(x, w, scales, bias, 2, 2, 0.05F);
    CHECK(y.size() == 2);
    CHECK(std::abs(y[0] - 1.4F) < 1e-5F);
    CHECK(std::abs(y[1] + 1.4F) < 1e-5F);

    threw = false;
    try {
        linear_i8(x, w, scales, bias, 2, 2, 0.F);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        linear_i8(x, w, scales, bias, std::numeric_limits<std::size_t>::max(), 2, 1.F);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    std::cout << "edgeai native tests passed\n";
    return 0;
}
