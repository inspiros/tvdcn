#pragma once

#include <ATen/ArrayRef.h>

namespace at {
    template<unsigned n, typename T>
    inline std::array<T, n> _ntuple(at::ArrayRef<T> x) {
        std::array<T, n> res;
        if (x.size() == n) {
            std::copy(x.begin(), x.end(), res.begin());
        } else if (x.size() == 1) {
            for (unsigned i = 0; i < n; i++)
                res[i] = x[0];
        } else {
            TORCH_CHECK(false,
                        "Expected a sequence of ",
                        n,
                        " elements. Got ",
                        x.size())
        }
        return res;
    }

    template<typename T>
    [[maybe_unused]] inline std::array<T, 1> _single(at::ArrayRef<T> x) {
        return _ntuple<1, T>(x);
    }

    template<typename T>
    [[maybe_unused]] inline std::array<T, 2> _pair(at::ArrayRef<T> x) {
        return _ntuple<2, T>(x);
    }

    template<typename T>
    [[maybe_unused]] inline std::array<T, 3> _triple(at::ArrayRef<T> x) {
        return _ntuple<3, T>(x);
    }

    template<typename T>
    [[maybe_unused]] inline std::array<T, 4> _quadruple(at::ArrayRef<T> x) {
        return _ntuple<4, T>(x);
    }
}
