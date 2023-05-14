#include <ATen/ATen.h>
#include "cuda_helpers.h"
#include "../utils/dispatch.h"

namespace tvdcn {
    namespace ops {
        namespace {
            constexpr float threadsFraction = 1.0;

            template<typename scalar_t>
            __device__ __forceinline__ scalar_t sample(
                    const at::GenericPackedTensorAccessor<scalar_t, 3> input,
                    const int b,
                    const int c,
                    const int width,
                    const int x) {
                return (0 <= x && x < width) ? input[b][c][x] : static_cast<scalar_t>(0);
            }

            template<typename scalar_t>
            __device__ __forceinline__ scalar_t interpolate_sample(
                    const at::GenericPackedTensorAccessor<scalar_t, 3> input,
                    const int b,
                    const int c,
                    const int width,
                    const scalar_t x) {
                if (x <= -1 || width <= x)
                    return 0;

                int x_l = floor(x);
                int x_h = x_l + 1;

                scalar_t dx_h = x - x_l;
                scalar_t dx_l = 1 - dx_h;

                bool valid_x_l = x_l >= 0;
                bool valid_x_h = x_h < width;

                scalar_t val = 0;
                if (valid_x_l) val += dx_l * input[b][c][x_l];
                if (valid_x_h) val += dx_h * input[b][c][x_h];
                return val;
            }

            template<typename scalar_t>
            __device__ __forceinline__ void insert(
                    at::GenericPackedTensorAccessor<scalar_t, 3> output,
                    const int b,
                    const int c,
                    const int width,
                    const int x,
                    const scalar_t val) {
                if (0 <= x && x < width)
                    gpuAtomicAdd(&output[b][c][x], val);
            }

            template<typename scalar_t>
            __device__ __forceinline__ void interpolate_insert(
                    at::GenericPackedTensorAccessor<scalar_t, 3> output,
                    const int b,
                    const int c,
                    const int width,
                    const scalar_t x,
                    const scalar_t val) {
                int x_l = floor(x);
                int x_h = x_l + 1;

                scalar_t dx_h = x - x_l;
                scalar_t dx_l = 1 - dx_h;

                bool valid_x_l = 0 <= x_l && x_l < width;
                bool valid_x_h = 0 <= x_h && x_h < width;

                if (valid_x_l) gpuAtomicAdd(&output[b][c][x_l], dx_l * val);
                if (valid_x_h) gpuAtomicAdd(&output[b][c][x_h], dx_h * val);
            }

            template<typename scalar_t>
            __device__ __forceinline__ scalar_t coordinate_weight(
                    const at::GenericPackedTensorAccessor<scalar_t, 3> input,
                    const int b,
                    const int c,
                    const int width,
                    const scalar_t x) {
                int x_l = floor(x);
                int x_h = x_l + 1;

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

        template<bool deformable, bool modulated, typename scalar_t>
        static __global__ void arr2col_kernel(
                const int n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 3> input,
                const at::GenericPackedTensorAccessor<scalar_t, 5> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 4> mask,
                const int width,
                const int weight_w,
                const int pad_w,
                const int stride_w,
                const int dilation_w,
                const int out_w,
                const int in_channels,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 4> columns) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                const int w = index % out_w;
                const int c = (index / out_w) % in_channels;
                const int b = index / (out_w * in_channels);

                const int offset_group_idx = c / c_per_offset_group;
                const int mask_group_idx = c / c_per_mask_group;

                for (int i = 0; i < weight_w; ++i) {
                    const int x = (w * stride_w - pad_w) + i * dilation_w;

                    const scalar_t val =
                            deformable ?
                            interpolate_sample(
                                    input, b, c, width,
                                    x + offset[b][offset_group_idx][i][0][w])
                                       : sample(input, b, c, width, x);

                    const scalar_t mask_val =
                            modulated ?
                            mask[b][mask_group_idx][i][w] : static_cast<scalar_t>(1);

                    columns[c][i][b][w] = val * mask_val;
                }
            }
        }

        void arr2col_cuda(
                const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int in_channels,
                const int width,
                const int weight_w,
                const int pad_w,
                const int stride_w,
                const int dilation_w,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &columns) {
            const int n_kernels = in_channels * out_w * batch_sz;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    input.scalar_type(), "arr2col_cuda", ([&] {
                auto columns_accessor = columns.generic_packed_accessor<scalar_t, 4>();
                TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                    arr2col_kernel<deformable, modulated><<<blocks, threads>>>(
                            n_kernels,
                            input.generic_packed_accessor<scalar_t, 3>(),
                            offset.generic_packed_accessor<scalar_t, 5>(),
                            mask.generic_packed_accessor<scalar_t, 4>(),
                            width,
                            weight_w,
                            pad_w,
                            stride_w,
                            dilation_w,
                            out_w,
                            in_channels,
                            c_per_offset_group,
                            c_per_mask_group,
                            columns_accessor);
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in arr2col_cuda: %s\n", cudaGetErrorString(err));
            }
        }

        template<bool deformable, bool modulated, typename scalar_t>
        static __global__ void col2arr_kernel(
                const int n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 4> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 5> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 4> mask,
                const int in_channels,
                const int width,
                const int weight_w,
                const int pad_w,
                const int stride_w,
                const int dilation_w,
                const int out_w,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 3> grad_input) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                const int i = index % weight_w;
                const int w = (index / weight_w) % out_w;
                const int c = (index / (weight_w * out_w)) % in_channels;
                const int b = (index / (weight_w * out_w * in_channels));

                const int offset_group_idx = c / c_per_offset_group;
                const int mask_group_idx = c / c_per_mask_group;

                const int x = (w * stride_w - pad_w) + i * dilation_w;

                const scalar_t mask_val =
                        modulated ?
                        mask[b][mask_group_idx][i][w] : static_cast<scalar_t>(1);

                const scalar_t val = columns[c][i][b][w] * mask_val;

                if (deformable)
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
                const int in_channels,
                const int width,
                const int weight_w,
                const int pad_w,
                const int stride_w,
                const int dilation_w,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_input) {
            const int n_kernels = batch_sz * in_channels * out_w * weight_w;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "col2arr_cuda", ([&] {
                auto grad_input_accessor = grad_input.generic_packed_accessor<scalar_t, 3>();
                TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                    col2arr_kernel<deformable, modulated><<<blocks, threads>>>(
                            n_kernels,
                            columns.generic_packed_accessor<scalar_t, 4>(),
                            offset.generic_packed_accessor<scalar_t, 5>(),
                            mask.generic_packed_accessor<scalar_t, 4>(),
                            in_channels,
                            width,
                            weight_w,
                            pad_w,
                            stride_w,
                            dilation_w,
                            out_w,
                            c_per_offset_group,
                            c_per_mask_group,
                            grad_input_accessor);
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in col2arr_cuda: %s\n", cudaGetErrorString(err));
            }
        }

        template<bool modulated, typename scalar_t>
        static __global__ void deform_conv1d_compute_grad_offset_kernel(
                const int n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 4> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 3> input,
                const at::GenericPackedTensorAccessor<scalar_t, 5> offset,
                const at::GenericPackedTensorAccessor<scalar_t, 4> mask,
                const int width,
                const int weight_w,
                const int pad_w,
                const int stride_w,
                const int dilation_w,
                const int out_w,
                const int n_offset_grps,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 5> grad_offset) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                const int i = index % weight_w;
                const int w = (index / weight_w) % out_w;
                const int g = (index / (weight_w * out_w)) % n_offset_grps;
                const int b = index / (weight_w * out_w * n_offset_grps);

                scalar_t grad_offset_val = 0;

                const int c_start = g * c_per_offset_group;
                const int c_end = c_start + c_per_offset_group;
                for (int c = c_start; c < c_end; ++c) {
                    const int mask_group_idx = c / c_per_mask_group;

                    const int x = (w * stride_w - pad_w) + i * dilation_w;

                    const scalar_t weight = coordinate_weight(
                            input, b, c, width,
                            x + offset[b][g][i][0][w]);

                    const scalar_t mask_val =
                            modulated ?
                            mask[b][mask_group_idx][i][w] : static_cast<scalar_t>(1);

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
                const int in_channels,
                const int width,
                const int weight_w,
                const int pad_w,
                const int stride_w,
                const int dilation_w,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_offset) {
            if (!deformable) return;
            const int n_kernels = batch_sz * n_offset_grps * out_w * weight_w;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv1d_compute_grad_offset_cuda", ([&] {
                auto grad_offset_accessor = grad_offset.generic_packed_accessor<scalar_t, 5>();
                TVDCN_DISPATCH_CONDITION(modulated, ([&] {
                    deform_conv1d_compute_grad_offset_kernel<modulated><<<blocks, threads>>>(
                            n_kernels,
                            columns.generic_packed_accessor<scalar_t, 4>(),
                            input.generic_packed_accessor<scalar_t, 3>(),
                            offset.generic_packed_accessor<scalar_t, 5>(),
                            mask.generic_packed_accessor<scalar_t, 4>(),
                            width,
                            weight_w,
                            pad_w,
                            stride_w,
                            dilation_w,
                            out_w,
                            n_offset_grps,
                            c_per_offset_group,
                            c_per_mask_group,
                            grad_offset_accessor);
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in deform_conv1d_compute_grad_offset_cuda: %s\n", cudaGetErrorString(err));
            }
        }

        template<bool deformable, typename scalar_t>
        static __global__ void deform_conv1d_compute_grad_mask_kernel(
                const int n_kernels,
                const at::GenericPackedTensorAccessor<scalar_t, 4> columns,
                const at::GenericPackedTensorAccessor<scalar_t, 3> input,
                const at::GenericPackedTensorAccessor<scalar_t, 5> offset,
                const int width,
                const int weight_w,
                const int pad_w,
                const int stride_w,
                const int dilation_w,
                const int out_w,
                const int n_mask_grps,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::GenericPackedTensorAccessor<scalar_t, 4> grad_mask) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                const int i = index % weight_w;
                const int w = (index / weight_w) % out_w;
                const int g = (index / (weight_w * out_w)) % n_mask_grps;
                const int b = index / (out_w * weight_w * n_mask_grps);

                scalar_t grad_mask_val = 0;

                const int c_start = g * c_per_mask_group;
                const int c_end = c_start + c_per_mask_group;
                for (int c = c_start; c < c_end; ++c) {
                    const int offset_group_idx = c / c_per_offset_group;

                    const int x = (w * stride_w - pad_w) + i * dilation_w;

                    const scalar_t val =
                            deformable ?
                            interpolate_sample(
                                    input, b, c, width,
                                    x + offset[b][offset_group_idx][i][0][w])
                                       : sample(input, b, c, width, x);

                    grad_mask_val += columns[c][i][b][w] * val;
                }

                grad_mask[b][g][i][w] = grad_mask_val;
            }
        }

        void deform_conv1d_compute_grad_mask_cuda(
                const at::Tensor &columns,
                const at::Tensor &input,
                const at::Tensor &offset,
                const int in_channels,
                const int width,
                const int weight_w,
                const int pad_w,
                const int stride_w,
                const int dilation_w,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_mask) {
            if (!modulated) return;
            const int n_kernels = batch_sz * n_mask_grps * out_w * weight_w;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv1d_compute_grad_mask_cuda", ([&] {
                auto grad_mask_accessor = grad_mask.generic_packed_accessor<scalar_t, 4>();
                TVDCN_DISPATCH_CONDITION(deformable, ([&] {
                    deform_conv1d_compute_grad_mask_kernel<deformable><<<blocks, threads>>>(
                            n_kernels,
                            columns.generic_packed_accessor<scalar_t, 4>(),
                            input.generic_packed_accessor<scalar_t, 3>(),
                            offset.generic_packed_accessor<scalar_t, 5>(),
                            width,
                            weight_w,
                            pad_w,
                            stride_w,
                            dilation_w,
                            out_w,
                            n_mask_grps,
                            c_per_offset_group,
                            c_per_mask_group,
                            grad_mask_accessor);
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in deform_conv1d_compute_grad_mask_cuda: %s\n", cudaGetErrorString(err));
            }
        }
    }
}
