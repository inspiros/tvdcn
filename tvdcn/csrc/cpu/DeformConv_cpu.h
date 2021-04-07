#pragma once

#include <cmath>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <TH/TH.h>

constexpr auto kMaxParallelImgs = 1;

static int get_greatest_divisor_below_bound(int n, int bound) {
    for (int k = std::min(n, bound); k > 1; --k) {
        if (n % k == 0) {
            return k;
        }
    }
    return 1;
}
