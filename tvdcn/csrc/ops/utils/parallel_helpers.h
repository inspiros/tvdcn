#pragma once

#include <ATen/ATen.h>

namespace tvdcn {
    namespace ops {
        constexpr auto kMaxParallelImgs = 32;

        static int get_greatest_divisor_below_bound(int n, int bound) {
            for (int k = std::min(n, bound); k > 1; --k) {
                if (n % k == 0) {
                    return k;
                }
            }
            return 1;
        }
    }
}
