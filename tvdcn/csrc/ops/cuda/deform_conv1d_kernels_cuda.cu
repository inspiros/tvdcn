#include <ATen/ATen.h>

#include "cuda_helpers.h"
#include "../utils/dispatch.h"

namespace tvdcn {
    namespace ops {
        namespace {
            constexpr float threadsFraction = 1.0;

            template<typename scalar_t, typename index_t>
            __device__ __forceinline__ scalar_t sample(
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> input,
                    const index_t b,
                    const index_t c,
                    const index_t width,
                    const index_t x) {
                return (0 <= x && x < width) ? input[b][c][x] : static_cast<scalar_t>(0);
            }

            template<typename scalar_t, typename index_t>
            __device__ __forceinline__ scalar_t interpolate_sample(
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> input,
                    const index_t b,
                    const index_t c,
                    const index_t width,
                    const scalar_t x) {
                if (x <= -1 || width <= x)
                    return 0;

                index_t x_l = floor(x);
                index_t x_h = x_l + 1;

                scalar_t dx_h = x - x_l;
                scalar_t dx_l = 1 - dx_h;

                bool valid_x_l = x_l >= 0;
                bool valid_x_h = x_h < width;

                scalar_t val = 0;
                if (valid_x_l) val += dx_l * input[b][c][x_l];
                if (valid_x_h) val += dx_h * input[b][c][x_h];
                return val;
            }

            template<typename scalar_t, typename index_t>
            __device__ __forceinline__ void insert(
                    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> output,
                    const index_t b,
                    const index_t c,
                    const index_t width,
                    const index_t x,
                    const scalar_t val) {
                if (0 <= x && x < width)
                    gpuAtomicAdd(&output[b][c][x], val);
            }

            template<typename scalar_t, typename index_t>
            __device__ __forceinline__ void interpolate_insert(
                    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> output,
                    const index_t b,
                    const index_t c,
                    const index_t width,
                    const scalar_t x,
                    const scalar_t val) {
                index_t x_l = floor(x);
                index_t x_h = x_l + 1;

                scalar_t dx_h = x - x_l;
                scalar_t dx_l = 1 - dx_h;

                bool valid_x_l = 0 <= x_l && x_l < width;
                bool valid_x_h = 0 <= x_h && x_h < width;

                if (valid_x_l) gpuAtomicAdd(&output[b][c][x_l], dx_l * val);
                if (valid_x_h) gpuAtomicAdd(&output[b][c][x_h], dx_h * val);
            }

            template<typename scalar_t, typename index_t>
            __device__ __forceinline__ scalar_t coordinate_weight(
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> input,
                    const index_t b,
                    const index_t c,
                    const index_t width,
                    const scalar_t x) {
                index_t x_l = floor(x);
                index_t x_h = x_l + 1;

                scalar_t dx_h = 1;
                scalar_t dx_l = -1;

                bool valid_x_l = 0 <= x_l && x_l < width;
                bool valid_x_h = 0 <= x_h && x_h < width;

                scalar_t val = 0;
                if (valid_x_l) val += dx_l * input[b][c][x_l];
                if (valid_x_h) val += dx_h * input[b][c][x_h];
                return val;
            }
        }

        template<bool deformable, bool modulated, typename scalar_t, typename index_t>
        static __global__ void arr2col_kernel(
                const index_t n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> input,
                const at::GenericPackedTensorAccessor<scalar_t, 5, at::RestrictPtrTraits, index_t> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> mask,
                const index_t width,
                const index_t weight_w,
                const index_t stride_w,
                const index_t pad_w,
                const index_t dilation_w,
                const index_t out_w,
                const index_t in_channels,
                const index_t c_per_offset_group,
                const index_t c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> columns) {
            CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                const index_t w = index % out_w;
                const index_t c = (index / out_w) % in_channels;
                const index_t b = index / (out_w * in_channels);

                const index_t offset_group_idx = c / c_per_offset_group;
                const index_t mask_group_idx = c / c_per_mask_group;

                for (index_t i = 0; i < weight_w; ++i) {
                    const index_t x = (w * stride_w - pad_w) + i * dilation_w;

                    scalar_t val, mask_val;
                    if constexpr (deformable)
                        val = interpolate_sample(
                                input, b, c, width,
                                x + offset[b][offset_group_idx][i][0][w]);
                    else
                        val = sample(input, b, c, width, x);

                    if constexpr (modulated)
                        mask_val = mask[b][mask_group_idx][i][w];
                    else
                        mask_val = static_cast<scalar_t>(1);

                    columns[c][i][b][w] = val * mask_val;
                }
            }
        }

        void arr2col_cuda(
                const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int64_t in_channels,
                const int64_t width,
                const int64_t weight_w,
                const int64_t stride_w,
                const int64_t pad_w,
                const int64_t dilation_w,
                const int64_t out_w,
                const int64_t batch_sz,
                const int64_t offset_groups,
                const int64_t mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &columns) {
            at::cuda::CUDAGuard device_guard(input.get_device());
            const int64_t n_kernels = (int64_t) batch_sz * in_channels * out_w;
            const int64_t c_per_offset_group = deformable ? in_channels / offset_groups : 1;
            const int64_t c_per_mask_group = modulated ? in_channels / mask_groups : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    input.scalar_type(), "arr2col_cuda", ([&] {
                TVDCN_DISPATCH_INDEX_TYPE2(n_kernels, columns.numel(), ([&] {
                    auto columns_accessor =
                            columns.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>();
                    TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                        arr2col_kernel<deformable, modulated, scalar_t, index_t><<<blocks, threads>>>(
                                n_kernels,
                                input.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                offset.generic_packed_accessor<scalar_t, 5, at::RestrictPtrTraits, index_t>(),
                                mask.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>(),
                                width,
                                weight_w,
                                stride_w,
                                pad_w,
                                dilation_w,
                                out_w,
                                in_channels,
                                c_per_offset_group,
                                c_per_mask_group,
                                columns_accessor);
                    }));
                }));
            }));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        template<bool deformable, bool modulated, typename scalar_t, typename index_t>
        static __global__ void col2arr_kernel(
                const index_t n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 5, at::RestrictPtrTraits, index_t> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> mask,
                const index_t in_channels,
                const index_t width,
                const index_t weight_w,
                const index_t stride_w,
                const index_t pad_w,
                const index_t dilation_w,
                const index_t out_w,
                const index_t c_per_offset_group,
                const index_t c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_input) {
            CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                const index_t i = index % weight_w;
                const index_t w = (index / weight_w) % out_w;
                const index_t c = (index / (weight_w * out_w)) % in_channels;
                const index_t b = (index / (weight_w * out_w * in_channels));

                const index_t offset_group_idx = c / c_per_offset_group;
                const index_t mask_group_idx = c / c_per_mask_group;

                const index_t x = (w * stride_w - pad_w) + i * dilation_w;

                scalar_t mask_val;
                if constexpr (modulated)
                    mask_val = mask[b][mask_group_idx][i][w];
                else
                    mask_val = static_cast<scalar_t>(1);

                scalar_t val = columns[c][i][b][w] * mask_val;

                if constexpr (deformable)
                    interpolate_insert(
                            grad_input, b, c, width,
                            x + offset[b][offset_group_idx][i][0][w],
                            val);
                else
                    insert(grad_input, b, c, width, x, val);
            }
        }

        void col2arr_cuda(
                const at::Tensor &columns,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int64_t in_channels,
                const int64_t width,
                const int64_t weight_w,
                const int64_t stride_w,
                const int64_t pad_w,
                const int64_t dilation_w,
                const int64_t out_w,
                const int64_t batch_sz,
                const int64_t offset_groups,
                const int64_t mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_input) {
            at::cuda::CUDAGuard device_guard(columns.get_device());
            const int64_t n_kernels = (int64_t) batch_sz * in_channels * out_w * weight_w;
            const int64_t c_per_offset_group = deformable ? in_channels / offset_groups : 1;
            const int64_t c_per_mask_group = modulated ? in_channels / mask_groups : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "col2arr_cuda", ([&] {
                TVDCN_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                    auto grad_input_accessor =
                            grad_input.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
                    TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                        col2arr_kernel<deformable, modulated, scalar_t, index_t><<<blocks, threads>>>(
                                n_kernels,
                                columns.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>(),
                                offset.generic_packed_accessor<scalar_t, 5, at::RestrictPtrTraits, index_t>(),
                                mask.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>(),
                                in_channels,
                                width,
                                weight_w,
                                stride_w,
                                pad_w,
                                dilation_w,
                                out_w,
                                c_per_offset_group,
                                c_per_mask_group,
                                grad_input_accessor);
                    }));
                }));
            }));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        template<bool modulated, typename scalar_t, typename index_t>
        static __global__ void deform_conv1d_compute_grad_offset_kernel(
                const index_t n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> input,
                const at::GenericPackedTensorAccessor<scalar_t, 5, at::RestrictPtrTraits, index_t> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> mask,
                const index_t width,
                const index_t weight_w,
                const index_t stride_w,
                const index_t pad_w,
                const index_t dilation_w,
                const index_t out_w,
                const index_t offset_groups,
                const index_t c_per_offset_group,
                const index_t c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 5, at::RestrictPtrTraits, index_t> grad_offset) {
            CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                const index_t i = index % weight_w;
                const index_t w = (index / weight_w) % out_w;
                const index_t g = (index / (weight_w * out_w)) % offset_groups;
                const index_t b = index / (weight_w * out_w * offset_groups);

                scalar_t grad_offset_val = 0;

                const index_t c_start = g * c_per_offset_group;
                const index_t c_end = c_start + c_per_offset_group;
                for (index_t c = c_start; c < c_end; ++c) {
                    const index_t mask_group_idx = c / c_per_mask_group;

                    const index_t x = (w * stride_w - pad_w) + i * dilation_w;

                    scalar_t weight = coordinate_weight(
                            input, b, c, width,
                            x + offset[b][g][i][0][w]);

                    scalar_t mask_val;
                    if constexpr (modulated)
                        mask_val = mask[b][mask_group_idx][i][w];
                    else
                        mask_val = static_cast<scalar_t>(1);

                    grad_offset_val += columns[c][i][b][w] * weight * mask_val;
                }

                grad_offset[b][g][i][0][w] = grad_offset_val;
            }
        }

        void deform_conv1d_compute_grad_offset_cuda(
                const at::Tensor &columns,
                const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int64_t in_channels,
                const int64_t width,
                const int64_t weight_w,
                const int64_t stride_w,
                const int64_t pad_w,
                const int64_t dilation_w,
                const int64_t out_w,
                const int64_t batch_sz,
                const int64_t offset_groups,
                const int64_t mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_offset) {
            if (!deformable) return;
            at::cuda::CUDAGuard device_guard(columns.get_device());
            const int64_t n_kernels = (int64_t) batch_sz * offset_groups * out_w * weight_w;
            const int64_t c_per_offset_group = deformable ? in_channels / offset_groups : 1;
            const int64_t c_per_mask_group = modulated ? in_channels / mask_groups : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv1d_compute_grad_offset_cuda", ([&] {
                TVDCN_DISPATCH_INDEX_TYPE2(n_kernels, columns.numel(), ([&] {
                    auto grad_offset_accessor =
                            grad_offset.generic_packed_accessor<scalar_t, 5, at::RestrictPtrTraits, index_t>();
                    TVDCN_DISPATCH_CONDITION(modulated, ([&] {
                        deform_conv1d_compute_grad_offset_kernel<modulated, scalar_t, index_t><<<blocks, threads>>>(
                                n_kernels,
                                columns.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>(),
                                input.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                offset.generic_packed_accessor<scalar_t, 5, at::RestrictPtrTraits, index_t>(),
                                mask.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>(),
                                width,
                                weight_w,
                                stride_w,
                                pad_w,
                                dilation_w,
                                out_w,
                                offset_groups,
                                c_per_offset_group,
                                c_per_mask_group,
                                grad_offset_accessor);
                    }));
                }));
            }));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        template<bool deformable, typename scalar_t, typename index_t>
        static __global__ void deform_conv1d_compute_grad_mask_kernel(
                const index_t n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> input,
                const at::GenericPackedTensorAccessor<scalar_t, 5, at::RestrictPtrTraits, index_t> offset,
                const index_t width,
                const index_t weight_w,
                const index_t stride_w,
                const index_t pad_w,
                const index_t dilation_w,
                const index_t out_w,
                const index_t mask_groups,
                const index_t c_per_offset_group,
                const index_t c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> grad_mask) {
            CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                const index_t i = index % weight_w;
                const index_t w = (index / weight_w) % out_w;
                const index_t g = (index / (weight_w * out_w)) % mask_groups;
                const index_t b = index / (out_w * weight_w * mask_groups);

                scalar_t grad_mask_val = 0;

                const index_t c_start = g * c_per_mask_group;
                const index_t c_end = c_start + c_per_mask_group;
                for (index_t c = c_start; c < c_end; ++c) {
                    const index_t offset_group_idx = c / c_per_offset_group;

                    const index_t x = (w * stride_w - pad_w) + i * dilation_w;

                    scalar_t val;
                    if constexpr (deformable)
                        val = interpolate_sample(
                                input, b, c, width,
                                x + offset[b][offset_group_idx][i][0][w]);
                    else
                        val = sample(input, b, c, width, x);

                    grad_mask_val += columns[c][i][b][w] * val;
                }

                grad_mask[b][g][i][w] = grad_mask_val;
            }
        }

        void deform_conv1d_compute_grad_mask_cuda(
                const at::Tensor &columns,
                const at::Tensor &input,
                const at::Tensor &offset,
                const int64_t in_channels,
                const int64_t width,
                const int64_t weight_w,
                const int64_t stride_w,
                const int64_t pad_w,
                const int64_t dilation_w,
                const int64_t out_w,
                const int64_t batch_sz,
                const int64_t offset_groups,
                const int64_t mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_mask) {
            if (!modulated) return;
            at::cuda::CUDAGuard device_guard(columns.get_device());
            const int64_t n_kernels = (int64_t) batch_sz * mask_groups * out_w * weight_w;
            const int64_t c_per_offset_group = deformable ? in_channels / offset_groups : 1;
            const int64_t c_per_mask_group = modulated ? in_channels / mask_groups : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv1d_compute_grad_mask_cuda", ([&] {
                TVDCN_DISPATCH_INDEX_TYPE2(n_kernels, columns.numel(), ([&] {
                    auto grad_mask_accessor =
                            grad_mask.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>();
                    TVDCN_DISPATCH_CONDITION(deformable, ([&] {
                        deform_conv1d_compute_grad_mask_kernel<deformable, scalar_t, index_t><<<blocks, threads>>>(
                                n_kernels,
                                columns.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>(),
                                input.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                offset.generic_packed_accessor<scalar_t, 5, at::RestrictPtrTraits, index_t>(),
                                width,
                                weight_w,
                                stride_w,
                                pad_w,
                                dilation_w,
                                out_w,
                                mask_groups,
                                c_per_offset_group,
                                c_per_mask_group,
                                grad_mask_accessor);
                    }));
                }));
            }));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
}
