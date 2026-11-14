#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>
#endif
#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <optional>
#include <vector>

namespace tvdcn::ops::stable_compat {
using torch::stable::Tensor;
using torch::headeronly::IntHeaderOnlyArrayRef;

// torch 2.14's dispatcher bridge cannot convert schema type Scalar. Use the
// stable C shims for these two scalar-bearing schemas instead. addmm_out with
// out=self preserves the exact legacy beta=alpha=1 GEMM accumulation.
// The backend is a template argument so _C never imports CUDA symbols.
template <bool Cuda = false>
inline Tensor addmm_(const Tensor& self, const Tensor& mat1, const Tensor& mat2) {
#ifdef USE_CUDA
    if constexpr (Cuda) {
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_addmm_out(
            self.get(), self.get(), mat1.get(), mat2.get(), 1.0, 1.0));
    } else
#endif
    {
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_addmm_out(
            self.get(), self.get(), mat1.get(), mat2.get(), 1.0, 1.0));
    }
    return self;
}

template <bool Cuda = false>
inline Tensor add_(Tensor self, const Tensor& other) {
    AtenTensorHandle result;
#ifdef USE_CUDA
    if constexpr (Cuda) {
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_add_Tensor(self.get(), other.get(), 1.0, &result));
    } else
#endif
    {
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_add_Tensor(self.get(), other.get(), 1.0, &result));
    }
    Tensor value(result);
    torch::stable::copy_(self, value);
    return self;
}

template <size_t N>
inline std::array<int64_t, N> ntuple(IntHeaderOnlyArrayRef value) {
    STD_TORCH_CHECK(value.size() == 1 || value.size() == N,
                    "Expected a sequence of ", N, " elements. Got ", value.size());
    std::array<int64_t, N> result;
    for (size_t i = 0; i < N; ++i)
        result[i] = value[value.size() == 1 ? 0 : i];
    return result;
}

// Largest index a kernel uses: its work-item count or the element count of any tensor it
// accesses (contiguous views, so offsets stay below numel). Selects 32- or 64-bit indices.
inline int64_t index_bound(int64_t n_kernels, std::initializer_list<std::reference_wrapper<const torch::stable::Tensor>> tensors) {
    int64_t bound = n_kernels;
    for (const torch::stable::Tensor &t : tensors)
        bound = std::max(bound, t.numel());
    return bound;
}

inline constexpr int64_t kMaxParallelImgs = 32;
inline int64_t get_greatest_divisor_below_bound(int64_t n, int64_t bound) {
    for (int64_t k = std::min(n, bound); k > 1; --k)
        if (n % k == 0) return k;
    return 1;
}

inline void check_tensors(const Tensor& input, const Tensor& weight,
                          const std::optional<Tensor>& offset,
                          const std::optional<Tensor>& mask,
                          const std::optional<Tensor>& bias, bool cuda) {
    STD_TORCH_CHECK(cuda ? input.is_cuda() : input.is_cpu(), "input is on the wrong device");
    auto check = [&](const Tensor& t) {
        STD_TORCH_CHECK(t.device() == input.device(), "Expected all tensors on the same device");
        STD_TORCH_CHECK(t.scalar_type() == input.scalar_type(), "Expected all tensors to have the same type");
    };
    check(weight);
    for (const auto* t : {&offset, &mask, &bias})
        if (t->has_value() && t->value().defined()) check(t->value());
}

// TensorAccessor borrows sizes/strides from the live Tensor; packed accessors
// copy them for launch arguments, exactly as the legacy accessors do.
template <typename T, size_t N>
inline torch::headeronly::HeaderOnlyTensorAccessor<T, N> accessor(const Tensor& t) {
    STD_TORCH_CHECK(t.dim() == N, "accessor rank mismatch");
    return {static_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data()};
}

#if defined(__CUDACC__)
template <typename T, size_t N, template<typename> class PtrTraits, typename Index>
inline torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<T, N, PtrTraits, Index>
packed_accessor(const Tensor& t) {
    STD_TORCH_CHECK(t.dim() == N, "packed accessor rank mismatch");
    return {static_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data()};
}
#endif
} // namespace tvdcn::ops::stable_compat
