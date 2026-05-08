// build_fixtures.cpp — emits ground-truth Cayley tables matching ganim's
// basis_blade_product_sign algorithm (sga.hpp:71-100).
//
// We reproduce the algorithm inline rather than calling ganim's function
// because ganim's `basis_blade_product_sign<Scalar, blade1, blade2, N>` takes
// blade values as compile-time template parameters; you cannot instantiate it
// 256 times in a runtime for-loop. The cross-language fixture is for the
// algorithm, not the symbol — both sides agreeing on numeric output is the
// correctness contract.
//
// Compile:  g++ -std=c++23 build_fixtures.cpp -o build_fixtures
// Run:      ./build_fixtures > pga3_cayley.json

#include <array>
#include <cstdio>
#include <cstdint>

int main() {
    constexpr std::array<std::int8_t, 4> pga3_metric = {1, 1, 1, 0};
    std::printf("[\n");
    bool first = true;
    for (std::uint64_t i = 0; i < 16; ++i) {
        for (std::uint64_t j = 0; j < 16; ++j) {
            // Inline replica of ganim's basis_blade_product_sign algorithm
            // (sga.hpp:71-100).
            bool s = false;
            std::uint64_t mask = 0;
            std::uint64_t b1 = i, b2 = j;
            bool null = false;
            while (b1 != 0) {
                std::uint64_t power1 =
                    std::uint64_t(1) << (63 - __builtin_clzll(b1));
                b2 &= power1 * 2 - 1;
                std::uint64_t b2_floor = (b2 == 0)
                    ? 0
                    : (std::uint64_t(1) << (63 - __builtin_clzll(b2)));
                if (power1 == b2_floor) {
                    int idx = __builtin_ctzll(power1);
                    if (pga3_metric[idx] == -1) s = !s;
                    else if (pga3_metric[idx] == 0) { null = true; break; }
                }
                b1 ^= power1;
                mask ^= power1 - 1;
            }
            int final_sign;
            if (null) {
                final_sign = 0;
            } else {
                bool parity = __builtin_popcountll(mask & j) % 2;
                s ^= parity;
                final_sign = s ? -1 : 1;
            }
            std::uint64_t result_blade = i ^ j;

            if (!first) std::printf(",\n");
            first = false;
            std::printf(
                "  {\"i\":%lu, \"j\":%lu, \"sign\":%d, \"blade\":%lu}",
                static_cast<unsigned long>(i),
                static_cast<unsigned long>(j),
                final_sign,
                static_cast<unsigned long>(result_blade)
            );
        }
    }
    std::printf("\n]\n");
    return 0;
}
