#include <ATen/ATen.h>
#include "cuda_helpers.h"
#include "../utils/dispatch.h"

namespace tvdcn {
    namespace ops {
        namespace {
            constexpr float threadsFraction = 0.75;

            template<typename scalar_t>
            __device__ __forceinline__ scalar_t sample(
                    const at::GenericPackedTensorAccessor<scalar_t, 4> input,
                    const int b,
                    const int c,
                    const int height,
                    const int width,
                    const int y,
                    const int x) {
                return (0 <= y && y < height && 0 <= x && x < width) ? input[b][c][y][x] : static_cast<scalar_t>(0);
            }

            template<typename scalar_t>
            __device__ __forceinline__ scalar_t interpolate_sample(
                    const at::GenericPackedTensorAccessor<scalar_t, 4> input,
                    const int b,
                    const int c,
                    const int height,
                    const int width,
                    const scalar_t y,
                    const scalar_t x) {
                if (y <= -1 || height <= y || x <= -1 || width <= x)
                    return 0;

                int y_l = floor(y);
                int y_h = y_l + 1;
                int x_l = floor(x);
                int x_h = x_l + 1;

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

            template<typename scalar_t>
            __device__ __forceinline__ void insert(
                    at::GenericPackedTensorAccessor<scalar_t, 4> output,
                    const int b,
                    const int c,
                    const int height,
                    const int width,
                    const int y,
                    const int x,
                    const scalar_t val) {
                if (0 <= y && y < height && 0 <= x && x < width)
                    gpuAtomicAdd(&output[b][c][y][x], val);
            }

            template<typename scalar_t>
            __device__ __forceinline__ void interpolate_insert(
                    at::GenericPackedTensorAccessor<scalar_t, 4> output,
                    const int b,
                    const int c,
                    const int height,
                    const int width,
                    const scalar_t y,
                    const scalar_t x,
                    const scalar_t val) {
                int y_l = floor(y);
                int y_h = y_l + 1;
                int x_l = floor(x);
                int x_h = x_l + 1;

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

            template<typename scalar_t>
            __device__ __forceinline__ scalar_t coordinate_weight(
                    const at::GenericPackedTensorAccessor<scalar_t, 4> input,
                    const int b,
                    const int c,
                    const int height,
                    const int width,
                    const scalar_t y,
                    const scalar_t x,
                    const int direction) {
                int y_l = floor(y);
                int y_h = y_l + 1;
                int x_l = floor(x);
                int x_h = x_l + 1;

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

        template<bool deformable, bool modulated, typename scalar_t>
        static __global__ void im2col_kernel(
                const int n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 4> input,
                const at::GenericPackedTensorAccessor<scalar_t, 7> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 6> mask,
                const int height,
                const int width,
                const int weight_h,
                const int weight_w,
                const int pad_h,
                const int pad_w,
                const int stride_h,
                const int stride_w,
                const int dilation_h,
                const int dilation_w,
                const int out_h,
                const int out_w,
                const int in_channels,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 6> columns) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                const int w = index % out_w;
                const int h = (index / out_w) % out_h;
                const int c = (index / (out_w * out_h)) % in_channels;
                const int b = index / (out_w * out_h * in_channels);

                const int offset_group_idx = c / c_per_offset_group;
                const int mask_group_idx = c / c_per_mask_group;

                for (int i = 0; i < weight_h; ++i) {
                    for (int j = 0; j < weight_w; ++j) {
                        const int y = (h * stride_h - pad_h) + i * dilation_h;
                        const int x = (w * stride_w - pad_w) + j * dilation_w;

                        const scalar_t val =
                                deformable ?
                                interpolate_sample(
                                        input, b, c, height, width,
                                        y + offset[b][offset_group_idx][i][j][0][h][w],
                                        x + offset[b][offset_group_idx][i][j][1][h][w])
                                           : sample(input, b, c, height, width, y, x);

                        const scalar_t mask_val =
                                modulated ? mask[b][mask_group_idx][i][j][h][w] : static_cast<scalar_t>(1);

                        columns[c][i][j][b][h][w] = val * mask_val;
                    }
                }
            }
        }

        void im2col_cuda(
                const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int in_channels,
                const int height,
                const int width,
                const int weight_h,
                const int weight_w,
                const int pad_h,
                const int pad_w,
                const int stride_h,
                const int stride_w,
                const int dilation_h,
                const int dilation_w,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &columns) {
            const int n_kernels = in_channels * out_h * out_w * batch_sz;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    input.scalar_type(), "im2col_cuda", ([&] {
                auto columns_accessor = columns.generic_packed_accessor<scalar_t, 6>();
                TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                    im2col_kernel<deformable, modulated><<<blocks, threads>>>(
                            n_kernels,
                            input.generic_packed_accessor<scalar_t, 4>(),
                            offset.generic_packed_accessor<scalar_t, 7>(),
                            mask.generic_packed_accessor<scalar_t, 6>(),
                            height,
                            width,
                            weight_h,
                            weight_w,
                            pad_h,
                            pad_w,
                            stride_h,
                            stride_w,
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

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in im2col_cuda: %s\n", cudaGetErrorString(err));
            }
        }

        template<bool deformable, bool modulated, typename scalar_t>
        static __global__ void col2im_kernel(
                const int n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 6> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 7> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 6> mask,
                const int in_channels,
                const int height,
                const int width,
                const int weight_h,
                const int weight_w,
                const int pad_h,
                const int pad_w,
                const int stride_h,
                const int stride_w,
                const int dilation_h,
                const int dilation_w,
                const int out_h,
                const int out_w,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 4> grad_input) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                const int j = index % weight_w;
                const int i = (index / weight_w) % weight_h;
                const int w = (index / (weight_w * weight_h)) % out_w;
                const int h = (index / (weight_w * weight_h * out_w)) % out_h;
                const int c = (index / (weight_w * weight_h * out_w * out_h)) % in_channels;
                const int b = (index / (weight_w * weight_h * out_w * out_h * in_channels));

                const int offset_group_idx = c / c_per_offset_group;
                const int mask_group_idx = c / c_per_mask_group;

                const int y = (h * stride_h - pad_h) + i * dilation_h;
                const int x = (w * stride_w - pad_w) + j * dilation_w;

                const scalar_t mask_val =
                        modulated ?
                        mask[b][mask_group_idx][i][j][h][w] : static_cast<scalar_t>(1);

                const scalar_t val = columns[c][i][j][b][h][w] * mask_val;

                if (deformable)
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
                const int in_channels,
                const int height,
                const int width,
                const int weight_h,
                const int weight_w,
                const int pad_h,
                const int pad_w,
                const int stride_h,
                const int stride_w,
                const int dilation_h,
                const int dilation_w,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_input) {
            const int n_kernels = batch_sz * in_channels * out_h * out_w * weight_h * weight_w;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "col2im_cuda", ([&] {
                auto grad_input_accessor = grad_input.generic_packed_accessor<scalar_t, 4>();
                TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                    col2im_kernel<deformable, modulated><<<blocks, threads>>>(
                            n_kernels,
                            columns.generic_packed_accessor<scalar_t, 6>(),
                            offset.generic_packed_accessor<scalar_t, 7>(),
                            mask.generic_packed_accessor<scalar_t, 6>(),
                            in_channels,
                            height,
                            width,
                            weight_h,
                            weight_w,
                            pad_h,
                            pad_w,
                            stride_h,
                            stride_w,
                            dilation_h,
                            dilation_w,
                            out_h,
                            out_w,
                            c_per_offset_group,
                            c_per_mask_group,
                            grad_input_accessor);
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in col2im_cuda: %s\n", cudaGetErrorString(err));
            }
        }

        template<bool modulated, typename scalar_t>
        static __global__ void deform_conv2d_compute_grad_offset_kernel(
                const int n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 6> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 4> input,
                const at::GenericPackedTensorAccessor<scalar_t, 7> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 6> mask,
                const int height,
                const int width,
                const int weight_h,
                const int weight_w,
                const int pad_h,
                const int pad_w,
                const int stride_h,
                const int stride_w,
                const int dilation_h,
                const int dilation_w,
                const int out_h,
                const int out_w,
                const int n_offset_grps,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 7> grad_offset) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                const int o = index % 2;
                const int j = (index / 2) % weight_w;
                const int i = (index / (2 * weight_w)) % weight_h;
                const int w = (index / (2 * weight_w * weight_h)) % out_w;
                const int h = (index / (2 * weight_w * weight_h * out_w)) % out_h;
                const int g = (index / (2 * weight_w * weight_h * out_w * out_h)) % n_offset_grps;
                const int b = index / (2 * weight_w * weight_h * out_w * out_h * n_offset_grps);

                scalar_t grad_offset_val = 0;

                const int c_start = g * c_per_offset_group;
                const int c_end = c_start + c_per_offset_group;
                for (int c = c_start; c < c_end; ++c) {
                    const int mask_group_idx = c / c_per_mask_group;

                    const int y = (h * stride_h - pad_h) + i * dilation_h;
                    const int x = (w * stride_w - pad_w) + j * dilation_w;

                    const scalar_t weight = coordinate_weight(
                            input, b, c, height, width,
                            y + offset[b][g][i][j][0][h][w],
                            x + offset[b][g][i][j][1][h][w],
                            o);

                    const scalar_t mask_val =
                            modulated ?
                            mask[b][mask_group_idx][i][j][h][w] : static_cast<scalar_t>(1);

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
                const int in_channels,
                const int height,
                const int width,
                const int weight_h,
                const int weight_w,
                const int pad_h,
                const int pad_w,
                const int stride_h,
                const int stride_w,
                const int dilation_h,
                const int dilation_w,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_offset) {
            if (!deformable) return;
            const int n_kernels = batch_sz * n_offset_grps * out_h * out_w * weight_h * weight_w * 2;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv2d_compute_grad_offset_cuda", ([&] {
                auto grad_offset_accessor = grad_offset.generic_packed_accessor<scalar_t, 7>();
                TVDCN_DISPATCH_CONDITION(modulated, ([&] {
                    deform_conv2d_compute_grad_offset_kernel<modulated><<<blocks, threads>>>(
                            n_kernels,
                            columns.generic_packed_accessor<scalar_t, 6>(),
                            input.generic_packed_accessor<scalar_t, 4>(),
                            offset.generic_packed_accessor<scalar_t, 7>(),
                            mask.generic_packed_accessor<scalar_t, 6>(),
                            height,
                            width,
                            weight_h,
                            weight_w,
                            pad_h,
                            pad_w,
                            stride_h,
                            stride_w,
                            dilation_h,
                            dilation_w,
                            out_h,
                            out_w,
                            n_offset_grps,
                            c_per_offset_group,
                            c_per_mask_group,
                            grad_offset_accessor);
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in deform_conv2d_compute_grad_offset_cuda: %s\n", cudaGetErrorString(err));
            }
        }

        template<bool deformable, typename scalar_t>
        static __global__ void deform_conv2d_compute_grad_mask_kernel(
                const int n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 6> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 4> input,
                const at::GenericPackedTensorAccessor<scalar_t, 7> offset,
                const int height,
                const int width,
                const int weight_h,
                const int weight_w,
                const int pad_h,
                const int pad_w,
                const int stride_h,
                const int stride_w,
                const int dilation_h,
                const int dilation_w,
                const int out_h,
                const int out_w,
                const int n_mask_grps,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 6> grad_mask) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                const int j = index % weight_w;
                const int i = (index / weight_w) % weight_h;
                const int w = (index / (weight_w * weight_h)) % out_w;
                const int h = (index / (weight_w * weight_h * out_w)) % out_h;
                const int g = (index / (weight_w * weight_h * out_w * out_h)) % n_mask_grps;
                const int b = index / (out_w * out_h * weight_w * weight_h * n_mask_grps);

                scalar_t grad_mask_val = 0;

                const int c_start = g * c_per_mask_group;
                const int c_end = c_start + c_per_mask_group;
                for (int c = c_start; c < c_end; ++c) {
                    const int offset_group_idx = c / c_per_offset_group;

                    const int y = (h * stride_h - pad_h) + i * dilation_h;
                    const int x = (w * stride_w - pad_w) + j * dilation_w;

                    const scalar_t val =
                            deformable ?
                            interpolate_sample(
                                    input, b, c, height, width,
                                    y + offset[b][offset_group_idx][i][j][0][h][w],
                                    x + offset[b][offset_group_idx][i][j][1][h][w])
                                       : sample(input, b, c, height, width, y, x);

                    grad_mask_val += columns[c][i][j][b][h][w] * val;
                }

                grad_mask[b][g][i][j][h][w] = grad_mask_val;
            }
        }

        void deform_conv2d_compute_grad_mask_cuda(
                const at::Tensor &columns,
                const at::Tensor &input,
                const at::Tensor &offset,
                const int in_channels,
                const int height,
                const int width,
                const int weight_h,
                const int weight_w,
                const int pad_h,
                const int pad_w,
                const int stride_h,
                const int stride_w,
                const int dilation_h,
                const int dilation_w,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_mask) {
            if (!modulated) return;
            const int n_kernels = batch_sz * n_mask_grps * out_h * out_w * weight_h * weight_w;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv2d_compute_grad_mask_cuda", ([&] {
                auto grad_mask_accessor = grad_mask.generic_packed_accessor<scalar_t, 6>();
                TVDCN_DISPATCH_CONDITION(deformable, ([&] {
                    deform_conv2d_compute_grad_mask_kernel<deformable><<<blocks, threads>>>(
                            n_kernels,
                            columns.generic_packed_accessor<scalar_t, 6>(),
                            input.generic_packed_accessor<scalar_t, 4>(),
                            offset.generic_packed_accessor<scalar_t, 7>(),
                            height,
                            width,
                            weight_h,
                            weight_w,
                            pad_h,
                            pad_w,
                            stride_h,
                            stride_w,
                            dilation_h,
                            dilation_w,
                            out_h,
                            out_w,
                            n_mask_grps,
                            c_per_offset_group,
                            c_per_mask_group,
                            grad_mask_accessor);
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in deform_conv2d_compute_grad_mask_cuda: %s\n", cudaGetErrorString(err));
            }
        }
    }
}
