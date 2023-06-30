#pragma once

#include <ATen/ATen.h>

namespace tvdcn {
    namespace ops {
        constexpr int64_t kMaxParallelImgs = 32;

        static inline int64_t get_greatest_divisor_below_bound(int64_t n, int64_t bound) {
            for (int64_t k = std::min(n, bound); k > 1; --k) {
                if (n % k == 0) {
                    return k;
                }
            }
            return 1;
        }
    }
}
