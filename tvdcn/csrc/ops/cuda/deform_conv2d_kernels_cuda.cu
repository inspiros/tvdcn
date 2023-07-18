#include <ATen/ATen.h>

#include "cuda_helpers.h"
#include "../utils/dispatch.h"

namespace tvdcn {
    namespace ops {
        namespace {
            constexpr float threadsFraction = 0.75;

            template<typename scalar_t, typename index_t>
            __device__ __forceinline__ scalar_t sample(
                    const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> input,
                    const index_t b,
                    const index_t c,
                    const index_t height,
                    const index_t width,
                    const index_t y,
                    const index_t x) {
                return (0 <= y && y < height && 0 <= x && x < width) ? input[b][c][y][x] : static_cast<scalar_t>(0);
            }

            template<typename scalar_t, typename index_t>
            __device__ __forceinline__ scalar_t interpolate_sample(
                    const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> input,
                    const index_t b,
                    const index_t c,
                    const index_t height,
                    const index_t width,
                    const scalar_t y,
                    const scalar_t x) {
                if (y <= -1 || height <= y || x <= -1 || width <= x)
                    return 0;

                index_t y_l = floor(y);
                index_t y_h = y_l + 1;
                index_t x_l = floor(x);
                index_t x_h = x_l + 1;

                scalar_t dy_h = y - y_l;
                scalar_t dx_h = x - x_l;
                scalar_t dy_l = 1 - dy_h;
                scalar_t dx_l = 1 - dx_h;

                bool valid_y_l = y_l >= 0;
                bool valid_y_h = y_h < height;
                bool valid_x_l = x_l >= 0;
                bool valid_x_h = x_h < width;

                scalar_t val = 0;
                if (valid_y_l && valid_x_l) val += dy_l * dx_l * input[b][c][y_l][x_l];
                if (valid_y_l && valid_x_h) val += dy_l * dx_h * input[b][c][y_l][x_h];
                if (valid_y_h && valid_x_l) val += dy_h * dx_l * input[b][c][y_h][x_l];
                if (valid_y_h && valid_x_h) val += dy_h * dx_h * input[b][c][y_h][x_h];
                return val;
            }

            template<typename scalar_t, typename index_t>
            __device__ __forceinline__ void insert(
                    at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> output,
                    const index_t b,
                    const index_t c,
                    const index_t height,
                    const index_t width,
                    const index_t y,
                    const index_t x,
                    const scalar_t val) {
                if (0 <= y && y < height && 0 <= x && x < width)
                    gpuAtomicAdd(&output[b][c][y][x], val);
            }

            template<typename scalar_t, typename index_t>
            __device__ __forceinline__ void interpolate_insert(
                    at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> output,
                    const index_t b,
                    const index_t c,
                    const index_t height,
                    const index_t width,
                    const scalar_t y,
                    const scalar_t x,
                    const scalar_t val) {
                index_t y_l = floor(y);
                index_t y_h = y_l + 1;
                index_t x_l = floor(x);
                index_t x_h = x_l + 1;

                scalar_t dy_h = y - y_l;
                scalar_t dx_h = x - x_l;
                scalar_t dy_l = 1 - dy_h;
                scalar_t dx_l = 1 - dx_h;

                bool valid_y_l = 0 <= y_l && y_l < height;
                bool valid_y_h = 0 <= y_h && y_h < height;
                bool valid_x_l = 0 <= x_l && x_l < width;
                bool valid_x_h = 0 <= x_h && x_h < width;

                if (valid_y_l && valid_x_l) gpuAtomicAdd(&output[b][c][y_l][x_l], dy_l * dx_l * val);
                if (valid_y_l && valid_x_h) gpuAtomicAdd(&output[b][c][y_l][x_h], dy_l * dx_h * val);
                if (valid_y_h && valid_x_l) gpuAtomicAdd(&output[b][c][y_h][x_l], dy_h * dx_l * val);
                if (valid_y_h && valid_x_h) gpuAtomicAdd(&output[b][c][y_h][x_h], dy_h * dx_h * val);
            }

            template<typename scalar_t, typename index_t>
            __device__ __forceinline__ scalar_t coordinate_weight(
                    const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> input,
                    const index_t b,
                    const index_t c,
                    const index_t height,
                    const index_t width,
                    const scalar_t y,
                    const scalar_t x,
                    const index_t direction) {
                index_t y_l = floor(y);
                index_t y_h = y_l + 1;
                index_t x_l = floor(x);
                index_t x_h = x_l + 1;

                scalar_t dy_h = (direction == 0) ? static_cast<scalar_t>(1) : y - y_l;
                scalar_t dy_l = (direction == 0) ? static_cast<scalar_t>(-1) : 1 - dy_h;
                scalar_t dx_h = (direction == 1) ? static_cast<scalar_t>(1) : x - x_l;
                scalar_t dx_l = (direction == 1) ? static_cast<scalar_t>(-1) : 1 - dx_h;

                bool valid_y_l = 0 <= y_l && y_l < height;
                bool valid_y_h = 0 <= y_h && y_h < height;
                bool valid_x_l = 0 <= x_l && x_l < width;
                bool valid_x_h = 0 <= x_h && x_h < width;

                scalar_t val = 0;
                if (valid_y_l && valid_x_l) val += dy_l * dx_l * input[b][c][y_l][x_l];
                if (valid_y_l && valid_x_h) val += dy_l * dx_h * input[b][c][y_l][x_h];
                if (valid_y_h && valid_x_l) val += dy_h * dx_l * input[b][c][y_h][x_l];
                if (valid_y_h && valid_x_h) val += dy_h * dx_h * input[b][c][y_h][x_h];
                return val;
            }
        }

        template<bool deformable, bool modulated, typename scalar_t, typename index_t>
        static __launch_bounds__(1024) __global__ void im2col_kernel(
                const index_t n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> input,
                const at::GenericPackedTensorAccessor<scalar_t, 7, at::RestrictPtrTraits, index_t> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, index_t> mask,
                const index_t height,
                const index_t width,
                const index_t weight_h,
                const index_t weight_w,
                const index_t stride_h,
                const index_t stride_w,
                const index_t pad_h,
                const index_t pad_w,
                const index_t dilation_h,
                const index_t dilation_w,
                const index_t out_h,
                const index_t out_w,
                const index_t in_channels,
                const index_t c_per_offset_group,
                const index_t c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, index_t> columns) {
            CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                const index_t w = index % out_w;
                const index_t h = (index / out_w) % out_h;
                const index_t c = (index / (out_w * out_h)) % in_channels;
                const index_t b = index / (out_w * out_h * in_channels);

                const index_t offset_group_idx = c / c_per_offset_group;
                const index_t mask_group_idx = c / c_per_mask_group;

                for (index_t i = 0; i < weight_h; ++i) {
                    for (index_t j = 0; j < weight_w; ++j) {
                        const index_t y = (h * stride_h - pad_h) + i * dilation_h;
                        const index_t x = (w * stride_w - pad_w) + j * dilation_w;

                        scalar_t val, mask_val;
                        if constexpr (deformable)
                            val = interpolate_sample(
                                    input, b, c, height, width,
                                    y + offset[b][offset_group_idx][i][j][0][h][w],
                                    x + offset[b][offset_group_idx][i][j][1][h][w]);
                        else
                            val = sample(input, b, c, height, width, y, x);

                        if constexpr (modulated)
                            mask_val = mask[b][mask_group_idx][i][j][h][w];
                        else
                            mask_val = static_cast<scalar_t>(1);

                        columns[c][i][j][b][h][w] = val * mask_val;
                    }
                }
            }
        }

        void im2col_cuda(
                const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int64_t in_channels,
                const int64_t height,
                const int64_t width,
                const int64_t weight_h,
                const int64_t weight_w,
                const int64_t stride_h,
                const int64_t stride_w,
                const int64_t pad_h,
                const int64_t pad_w,
                const int64_t dilation_h,
                const int64_t dilation_w,
                const int64_t out_h,
                const int64_t out_w,
                const int64_t batch_sz,
                const int64_t offset_groups,
                const int64_t mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &columns) {
            at::cuda::CUDAGuard device_guard(input.get_device());
            const int64_t n_kernels = (int64_t) batch_sz * in_channels * out_h * out_w;
            const int64_t c_per_offset_group = deformable ? in_channels / offset_groups : 1;
            const int64_t c_per_mask_group = modulated ? in_channels / mask_groups : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    input.scalar_type(), "im2col_cuda", ([&] {
                TVDCN_DISPATCH_INDEX_TYPE2(n_kernels, columns.numel(), ([&] {
                    auto columns_accessor =
                            columns.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, index_t>();
                    TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                        im2col_kernel<deformable, modulated, scalar_t, index_t><<<blocks, threads>>>(
                                n_kernels,
                                input.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>(),
                                offset.generic_packed_accessor<scalar_t, 7, at::RestrictPtrTraits, index_t>(),
                                mask.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, index_t>(),
                                height,
                                width,
                                weight_h,
                                weight_w,
                                stride_h,
                                stride_w,
                                pad_h,
                                pad_w,
                                dilation_h,
                                dilation_w,
                                out_h,
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
        static __launch_bounds__(1024) __global__ void col2im_kernel(
                const index_t n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, index_t> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 7, at::RestrictPtrTraits, index_t> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, index_t> mask,
                const index_t in_channels,
                const index_t height,
                const index_t width,
                const index_t weight_h,
                const index_t weight_w,
                const index_t stride_h,
                const index_t stride_w,
                const index_t pad_h,
                const index_t pad_w,
                const index_t dilation_h,
                const index_t dilation_w,
                const index_t out_h,
                const index_t out_w,
                const index_t c_per_offset_group,
                const index_t c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> grad_input) {
            CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                const index_t j = index % weight_w;
                const index_t i = (index / weight_w) % weight_h;
                const index_t w = (index / (weight_w * weight_h)) % out_w;
                const index_t h = (index / (weight_w * weight_h * out_w)) % out_h;
                const index_t c = (index / (weight_w * weight_h * out_w * out_h)) % in_channels;
                const index_t b = (index / (weight_w * weight_h * out_w * out_h * in_channels));

                const index_t offset_group_idx = c / c_per_offset_group;
                const index_t mask_group_idx = c / c_per_mask_group;

                const index_t y = (h * stride_h - pad_h) + i * dilation_h;
                const index_t x = (w * stride_w - pad_w) + j * dilation_w;

                scalar_t mask_val;
                if constexpr (modulated)
                    mask_val = mask[b][mask_group_idx][i][j][h][w];
                else
                    mask_val = static_cast<scalar_t>(1);

                scalar_t val = columns[c][i][j][b][h][w] * mask_val;

                if constexpr (deformable)
                    interpolate_insert(
                            grad_input, b, c, height, width,
                            y + offset[b][offset_group_idx][i][j][0][h][w],
                            x + offset[b][offset_group_idx][i][j][1][h][w],
                            val);
                else
                    insert(grad_input, b, c, height, width, y, x, val);
            }
        }

        void col2im_cuda(
                const at::Tensor &columns,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int64_t in_channels,
                const int64_t height,
                const int64_t width,
                const int64_t weight_h,
                const int64_t weight_w,
                const int64_t stride_h,
                const int64_t stride_w,
                const int64_t pad_h,
                const int64_t pad_w,
                const int64_t dilation_h,
                const int64_t dilation_w,
                const int64_t out_h,
                const int64_t out_w,
                const int64_t batch_sz,
                const int64_t offset_groups,
                const int64_t mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_input) {
            at::cuda::CUDAGuard device_guard(columns.get_device());
            const int64_t n_kernels = (int64_t) batch_sz * in_channels * out_h * out_w * weight_h * weight_w;
            const int64_t c_per_offset_group = deformable ? in_channels / offset_groups : 1;
            const int64_t c_per_mask_group = modulated ? in_channels / mask_groups : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "col2im_cuda", ([&] {
                TVDCN_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                    auto grad_input_accessor =
                            grad_input.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>();
                    TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                        col2im_kernel<deformable, modulated, scalar_t, index_t><<<blocks, threads>>>(
                                n_kernels,
                                columns.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, index_t>(),
                                offset.generic_packed_accessor<scalar_t, 7, at::RestrictPtrTraits, index_t>(),
                                mask.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, index_t>(),
                                in_channels,
                                height,
                                width,
                                weight_h,
                                weight_w,
                                stride_h,
                                stride_w,
                                pad_h,
                                pad_w,
                                dilation_h,
                                dilation_w,
                                out_h,
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
        static __launch_bounds__(1024) __global__ void deform_conv2d_compute_grad_offset_kernel(
                const index_t n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, index_t> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> input,
                const at::GenericPackedTensorAccessor<scalar_t, 7, at::RestrictPtrTraits, index_t> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, index_t> mask,
                const index_t height,
                const index_t width,
                const index_t weight_h,
                const index_t weight_w,
                const index_t stride_h,
                const index_t stride_w,
                const index_t pad_h,
                const index_t pad_w,
                const index_t dilation_h,
                const index_t dilation_w,
                const index_t out_h,
                const index_t out_w,
                const index_t offset_groups,
                const index_t c_per_offset_group,
                const index_t c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 7, at::RestrictPtrTraits, index_t> grad_offset) {
            CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                const index_t o = index % 2;
                const index_t j = (index / 2) % weight_w;
                const index_t i = (index / (2 * weight_w)) % weight_h;
                const index_t w = (index / (2 * weight_w * weight_h)) % out_w;
                const index_t h = (index / (2 * weight_w * weight_h * out_w)) % out_h;
                const index_t g = (index / (2 * weight_w * weight_h * out_w * out_h)) % offset_groups;
                const index_t b = index / (2 * weight_w * weight_h * out_w * out_h * offset_groups);

                scalar_t grad_offset_val = 0;

                const index_t c_start = g * c_per_offset_group;
                const index_t c_end = c_start + c_per_offset_group;
                for (index_t c = c_start; c < c_end; ++c) {
                    const index_t mask_group_idx = c / c_per_mask_group;

                    const index_t y = (h * stride_h - pad_h) + i * dilation_h;
                    const index_t x = (w * stride_w - pad_w) + j * dilation_w;

                    scalar_t weight = coordinate_weight(
                            input, b, c, height, width,
                            y + offset[b][g][i][j][0][h][w],
                            x + offset[b][g][i][j][1][h][w],
                            o);

                    scalar_t mask_val;
                    if constexpr (modulated)
                        mask_val = mask[b][mask_group_idx][i][j][h][w];
                    else
                        mask_val = static_cast<scalar_t>(1);

                    grad_offset_val += columns[c][i][j][b][h][w] * weight * mask_val;
                }

                grad_offset[b][g][i][j][o][h][w] = grad_offset_val;
            }
        }

        void deform_conv2d_compute_grad_offset_cuda(
                const at::Tensor &columns,
                const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int64_t in_channels,
                const int64_t height,
                const int64_t width,
                const int64_t weight_h,
                const int64_t weight_w,
                const int64_t stride_h,
                const int64_t stride_w,
                const int64_t pad_h,
                const int64_t pad_w,
                const int64_t dilation_h,
                const int64_t dilation_w,
                const int64_t out_h,
                const int64_t out_w,
                const int64_t batch_sz,
                const int64_t offset_groups,
                const int64_t mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_offset) {
            if (!deformable) return;
            at::cuda::CUDAGuard device_guard(columns.get_device());
            const int64_t n_kernels = (int64_t) batch_sz * offset_groups * out_h * out_w * weight_h * weight_w * 2;
            const int64_t c_per_offset_group = deformable ? in_channels / offset_groups : 1;
            const int64_t c_per_mask_group = modulated ? in_channels / mask_groups : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv2d_compute_grad_offset_cuda", ([&] {
                TVDCN_DISPATCH_INDEX_TYPE2(n_kernels, columns.numel(), ([&] {
                    auto grad_offset_accessor =
                            grad_offset.generic_packed_accessor<scalar_t, 7, at::RestrictPtrTraits, index_t>();
                    TVDCN_DISPATCH_CONDITION(modulated, ([&] {
                        deform_conv2d_compute_grad_offset_kernel<modulated, scalar_t, index_t><<<blocks, threads>>>(
                                n_kernels,
                                columns.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, index_t>(),
                                input.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>(),
                                offset.generic_packed_accessor<scalar_t, 7, at::RestrictPtrTraits, index_t>(),
                                mask.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, index_t>(),
                                height,
                                width,
                                weight_h,
                                weight_w,
                                stride_h,
                                stride_w,
                                pad_h,
                                pad_w,
                                dilation_h,
                                dilation_w,
                                out_h,
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
        static __launch_bounds__(1024) __global__ void deform_conv2d_compute_grad_mask_kernel(
                const index_t n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, index_t> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> input,
                const at::GenericPackedTensorAccessor<scalar_t, 7, at::RestrictPtrTraits, index_t> offset,
                const index_t height,
                const index_t width,
                const index_t weight_h,
                const index_t weight_w,
                const index_t stride_h,
                const index_t stride_w,
                const index_t pad_h,
                const index_t pad_w,
                const index_t dilation_h,
                const index_t dilation_w,
                const index_t out_h,
                const index_t out_w,
                const index_t mask_groups,
                const index_t c_per_offset_group,
                const index_t c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, index_t> grad_mask) {
            CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                const index_t j = index % weight_w;
                const index_t i = (index / weight_w) % weight_h;
                const index_t w = (index / (weight_w * weight_h)) % out_w;
                const index_t h = (index / (weight_w * weight_h * out_w)) % out_h;
                const index_t g = (index / (weight_w * weight_h * out_w * out_h)) % mask_groups;
                const index_t b = index / (out_w * out_h * weight_w * weight_h * mask_groups);

                scalar_t grad_mask_val = 0;

                const index_t c_start = g * c_per_mask_group;
                const index_t c_end = c_start + c_per_mask_group;
                for (index_t c = c_start; c < c_end; ++c) {
                    const index_t offset_group_idx = c / c_per_offset_group;

                    const index_t y = (h * stride_h - pad_h) + i * dilation_h;
                    const index_t x = (w * stride_w - pad_w) + j * dilation_w;

                    scalar_t val;
                    if constexpr (deformable)
                        val = interpolate_sample(
                                input, b, c, height, width,
                                y + offset[b][offset_group_idx][i][j][0][h][w],
                                x + offset[b][offset_group_idx][i][j][1][h][w]);
                    else
                        val = sample(input, b, c, height, width, y, x);

                    grad_mask_val += columns[c][i][j][b][h][w] * val;
                }

                grad_mask[b][g][i][j][h][w] = grad_mask_val;
            }
        }

        void deform_conv2d_compute_grad_mask_cuda(
                const at::Tensor &columns,
                const at::Tensor &input,
                const at::Tensor &offset,
                const int64_t in_channels,
                const int64_t height,
                const int64_t width,
                const int64_t weight_h,
                const int64_t weight_w,
                const int64_t stride_h,
                const int64_t stride_w,
                const int64_t pad_h,
                const int64_t pad_w,
                const int64_t dilation_h,
                const int64_t dilation_w,
                const int64_t out_h,
                const int64_t out_w,
                const int64_t batch_sz,
                const int64_t offset_groups,
                const int64_t mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_mask) {
            if (!modulated) return;
            at::cuda::CUDAGuard device_guard(columns.get_device());
            const int64_t n_kernels = (int64_t) batch_sz * mask_groups * out_h * out_w * weight_h * weight_w;
            const int64_t c_per_offset_group = deformable ? in_channels / offset_groups : 1;
            const int64_t c_per_mask_group = modulated ? in_channels / mask_groups : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv2d_compute_grad_mask_cuda", ([&] {
                TVDCN_DISPATCH_INDEX_TYPE2(n_kernels, columns.numel(), ([&] {
                    auto grad_mask_accessor =
                            grad_mask.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, index_t>();
                    TVDCN_DISPATCH_CONDITION(deformable, ([&] {
                        deform_conv2d_compute_grad_mask_kernel<deformable, scalar_t, index_t><<<blocks, threads>>>(
                                n_kernels,
                                columns.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, index_t>(),
                                input.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>(),
                                offset.generic_packed_accessor<scalar_t, 7, at::RestrictPtrTraits, index_t>(),
                                height,
                                width,
                                weight_h,
                                weight_w,
                                stride_h,
                                stride_w,
                                pad_h,
                                pad_w,
                                dilation_h,
                                dilation_w,
                                out_h,
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
