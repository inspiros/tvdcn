#include <ATen/ATen.h>
#include "cuda_helpers.h"
#include "../utils/dispatch.h"

namespace tvdcn {
    namespace ops {
        namespace {
            constexpr float threadsFraction = 1.0;

            template<typename scalar_t>
            static __device__ __forceinline__ scalar_t sample(
                    const scalar_t *input,
                    const int width,
                    const int x) {
                return (0 <= x && x < width) ? input[x] : static_cast<scalar_t>(0);
            }

            template<typename scalar_t>
            static __device__ scalar_t interpolate_sample(
                    const scalar_t *input,
                    const int width,
                    const scalar_t x) {
                if (x <= -1 || width <= x) {
                    return 0;
                }

                int x_l = floor(x);
                int x_h = x_l + 1;

                scalar_t dx_h = x - x_l;
                scalar_t dx_l = 1 - dx_h;

                bool valid_x_l = x_l >= 0;
                bool valid_x_h = x_h < width;

                scalar_t val = 0;
                if (valid_x_l) val += dx_l * input[x_l];
                if (valid_x_h) val += dx_h * input[x_h];

                return val;
            }

            template<typename scalar_t>
            static __device__ __forceinline__ void insert(
                    scalar_t *output,
                    const int width,
                    const int x,
                    const scalar_t val) {
                if (0 <= x && x < width)
                    atomicAdd(output + x, val);
            }

            template<typename scalar_t>
            static __device__ void interpolate_insert(
                    scalar_t *output,
                    const int width,
                    const scalar_t x,
                    const scalar_t val) {
                int x_l = floor(x);
                int x_h = x_l + 1;

                scalar_t dx_h = x - x_l;
                scalar_t dx_l = 1 - dx_h;

                bool valid_x_l = 0 <= x_l && x_l < width;
                bool valid_x_h = 0 <= x_h && x_h < width;

                if (valid_x_l) atomicAdd(output + x_l, dx_l * val);
                if (valid_x_h) atomicAdd(output + x_h, dx_h * val);
            }

            template<typename scalar_t>
            static __device__ scalar_t linear_coordinate_weight(
                    const scalar_t *input,
                    const int width,
                    const scalar_t x) {
                int x_l = floor(x);
                int x_h = x_l + 1;

                scalar_t dx_h = 1;
                scalar_t dx_l = -1;

                bool valid_x_l = 0 <= x_l && x_l < width;
                bool valid_x_h = 0 <= x_h && x_h < width;

                scalar_t val = 0;
                if (valid_x_l) val += dx_l * input[x_l];
                if (valid_x_h) val += dx_h * input[x_h];

                return val;
            }
        }

        template<bool deformable, bool modulated, typename scalar_t>
        static __global__ void arr2col_kernel(
                const int n_kernels,
                const int c_per_offset_grp,
                const int c_per_mask_grp,
                const scalar_t *input,
                const scalar_t *offset,
                const scalar_t *mask,
                const int width,
                const int weight_w,
                const int pad_w,
                const int stride_w,
                const int dilation_w,
                const int out_w,
                const int batch_sz,
                const int in_channels,
                const int n_offset_grps,
                const int n_mask_grps,
                scalar_t *columns) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                int out_x = index % out_w;
                int out_b = (index / out_w) % batch_sz;
                int in_c = index / (out_w * batch_sz);
                int out_c = in_c * weight_w;

                int offset_grp = in_c / c_per_offset_grp;
                int mask_grp = in_c / c_per_mask_grp;

                int columns_ptr = (out_c * (batch_sz * out_w) + out_b * out_w + out_x);
                int input_ptr = (out_b * (in_channels * width) + in_c * width);
                int offset_ptr = (out_b * n_offset_grps + offset_grp) * weight_w * out_w;
                int mask_ptr = (out_b * n_mask_grps + mask_grp) * weight_w * out_w;

                for (int i = 0; i < weight_w; ++i) {
                    int mask_idx = i;
                    int offset_idx = mask_idx;

                    const scalar_t offset_w = deformable ?
                                              offset[offset_ptr + offset_idx * out_w + out_x]
                                                         : static_cast<scalar_t>(0);
                    const scalar_t x = (out_x * stride_w - pad_w) + i * dilation_w + offset_w;
                    const scalar_t val = deformable ?
                                         interpolate_sample(input + input_ptr, width, x)
                                                    : sample(input + input_ptr, width, static_cast<int>(x));

                    const scalar_t mask_val = modulated ?
                                              mask[mask_ptr + mask_idx * out_w + out_x] : static_cast<scalar_t>(1);

                    columns[columns_ptr] = val * mask_val;
                    columns_ptr += batch_sz * out_w;
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
            const int c_per_offset_grp = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_grp = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    input.scalar_type(), "arr2col_cuda", ([&] {
                TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                    arr2col_kernel<deformable, modulated><<<blocks, threads>>>(
                            n_kernels,
                            c_per_offset_grp,
                            c_per_mask_grp,
                            input.data_ptr<scalar_t>(),
                            offset.data_ptr<scalar_t>(),
                            mask.data_ptr<scalar_t>(),
                            width,
                            weight_w,
                            pad_w,
                            stride_w,
                            dilation_w,
                            out_w,
                            batch_sz,
                            in_channels,
                            n_offset_grps,
                            n_mask_grps,
                            columns.data_ptr<scalar_t>());
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
                const int c_per_offset_grp,
                const int c_per_mask_grp,
                const scalar_t *columns,
                const scalar_t *offset,
                const scalar_t *mask,
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
                scalar_t *grad_input) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                int out_x = index % out_w;
                int b = (index / out_w) % batch_sz;
                int i = (index / (out_w * batch_sz)) % weight_w;
                int c = index / (out_w * batch_sz * weight_w);

                int offset_grp = c / c_per_offset_grp;
                int mask_grp = c / c_per_mask_grp;

                int offset_ptr = (b * n_offset_grps + offset_grp) * weight_w * out_w;
                int mask_ptr = (b * n_mask_grps + mask_grp) * weight_w * out_w;

                int mask_idx = i;
                int offset_idx = mask_idx;
                int offset_w_ptr = offset_idx * out_w + out_x;

                const scalar_t offset_w = deformable ?
                                          offset[offset_ptr + offset_w_ptr] : static_cast<scalar_t>(0);
                const scalar_t x = (out_x * stride_w - pad_w) + i * dilation_w + offset_w;

                const scalar_t mask_val = modulated ?
                                          mask[mask_ptr + mask_idx * out_w + out_x] : static_cast<scalar_t>(1);

                const scalar_t val = columns[index] * mask_val;

                int grad_input_ptr = (b * in_channels + c) * width;
                if (deformable)
                    interpolate_insert(grad_input + grad_input_ptr, width, x, val);
                else
                    insert(grad_input + grad_input_ptr, width, x, val);
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
            const int n_kernels = in_channels * weight_w * out_w * batch_sz;
            const int c_per_offset_grp = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_grp = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "col2arr_cuda", ([&] {
                TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                    col2arr_kernel<deformable, modulated><<<blocks, threads>>>(
                            n_kernels,
                            c_per_offset_grp,
                            c_per_mask_grp,
                            columns.data_ptr<scalar_t>(),
                            offset.data_ptr<scalar_t>(),
                            mask.data_ptr<scalar_t>(),
                            in_channels,
                            width,
                            weight_w,
                            pad_w,
                            stride_w,
                            dilation_w,
                            out_w,
                            batch_sz,
                            n_offset_grps,
                            n_mask_grps,
                            grad_input.data_ptr<scalar_t>());
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
                const int n_offset_kernels,
                const int offset_channels,
                const int c_per_offset_grp,
                const int c_per_mask_grp,
                const scalar_t *columns,
                const scalar_t *input,
                const scalar_t *offset,
                const scalar_t *mask,
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
                scalar_t *grad_offset) {
            CUDA_1D_KERNEL_LOOP(index, n_offset_kernels) {
                scalar_t grad_offset_val = 0;

                int w = index % out_w;
                int c = (index / out_w) % offset_channels;
                int b = index / (out_w * offset_channels);

                int offset_grp = c / weight_w;

                int col_ptr = offset_grp * c_per_offset_grp * weight_w * batch_sz * out_w;
                int input_ptr = (b * n_offset_grps + offset_grp) * c_per_offset_grp * width;
                int offset_ptr = (b * n_offset_grps + offset_grp) * weight_w * out_w;

                int offset_c = c - offset_grp * weight_w;

                const int c_bound = c_per_offset_grp * weight_w;
                const int col_step = weight_w;
                for (int col_c = offset_c; col_c < c_bound; col_c += col_step) {
                    int col_pos = ((col_c * batch_sz + b) * out_w) + w;
                    int in_c = (col_ptr + col_pos) * in_channels / n_kernels;

                    int mask_grp = in_c / c_per_mask_grp;
                    int mask_ptr = (b * n_mask_grps + mask_grp) * weight_w * out_w;

                    int out_x = col_pos % out_w;
                    int i = (col_pos / (out_w * batch_sz)) % weight_w;

                    int mask_idx = i;
                    int offset_idx = mask_idx;

                    const scalar_t offset_w = offset[offset_ptr + offset_idx * out_w + out_x];
                    const scalar_t x = (out_x * stride_w - pad_w) + i * dilation_w + offset_w;
                    const scalar_t weight = linear_coordinate_weight(input + input_ptr, width, x);

                    const scalar_t mask_val = modulated ?
                                              mask[mask_ptr + mask_idx * out_w + out_x] : static_cast<scalar_t>(1);

                    grad_offset_val += columns[col_ptr + col_pos] * weight * mask_val;
                    input_ptr += width;
                }

                grad_offset[index] = grad_offset_val;
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
            const int n_kernels = out_w * weight_w * in_channels * batch_sz;
            const int n_offset_kernels = out_w * weight_w * n_offset_grps * batch_sz;
            const int offset_channels = weight_w * n_offset_grps;
            const int c_per_offset_grp = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_grp = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv1d_compute_grad_offset_cuda", ([&] {
                TVDCN_DISPATCH_CONDITION(modulated, ([&] {
                    deform_conv1d_compute_grad_offset_kernel<modulated><<<blocks, threads>>>(
                            n_kernels,
                            n_offset_kernels,
                            offset_channels,
                            c_per_offset_grp,
                            c_per_mask_grp,
                            columns.data_ptr<scalar_t>(),
                            input.data_ptr<scalar_t>(),
                            offset.data_ptr<scalar_t>(),
                            mask.data_ptr<scalar_t>(),
                            in_channels,
                            width,
                            weight_w,
                            pad_w,
                            stride_w,
                            dilation_w,
                            out_w,
                            batch_sz,
                            n_offset_grps,
                            n_mask_grps,
                            grad_offset.data_ptr<scalar_t>());
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
                const int n_mask_kernels,
                const int mask_channels,
                const int c_per_offset_grp,
                const int c_per_mask_grp,
                const scalar_t *columns,
                const scalar_t *input,
                const scalar_t *offset,
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
                scalar_t *grad_mask) {
            CUDA_1D_KERNEL_LOOP(index, n_mask_kernels) {
                scalar_t grad_mask_val = 0;

                int w = index % out_w;
                int c = (index / out_w) % mask_channels;
                int b = index / (out_w * mask_channels);

                int mask_grp = c / weight_w;

                int col_ptr = mask_grp * c_per_mask_grp * weight_w * batch_sz * out_w;
                int input_ptr = (b * n_mask_grps + mask_grp) * c_per_mask_grp * width;
                int mask_ptr = (b * n_mask_grps + mask_grp) * weight_w * out_w;

                int mask_c = c - mask_grp * weight_w;

                const int c_bound = c_per_mask_grp * weight_w;
                const int col_step = weight_w;
                for (int col_c = mask_c; col_c < c_bound; col_c += col_step) {
                    int col_pos = ((col_c * batch_sz + b) * out_w) + w;
                    int in_c = (col_ptr + col_pos) * in_channels / n_kernels;

                    int offset_grp = in_c / c_per_offset_grp;
                    int offset_ptr = (b * n_offset_grps + offset_grp) * weight_w * out_w;

                    int out_x = col_pos % out_w;
                    int i = (col_pos / (out_w * batch_sz)) % weight_w;

                    int mask_idx = i;
                    int offset_idx = mask_idx;
                    const scalar_t offset_w = deformable ?
                                              offset[offset_ptr + offset_idx * out_w + out_x]
                                                         : static_cast<scalar_t>(0);
                    const scalar_t x = (out_x * stride_w - pad_w) + i * dilation_w + offset_w;
                    const scalar_t val = deformable ?
                                         interpolate_sample(input + input_ptr, width, x)
                                                    : sample(input + input_ptr, width, static_cast<int>(x));

                    grad_mask_val += columns[col_ptr + col_pos] * val;
                    input_ptr += width;
                }

                grad_mask[mask_ptr + mask_c * out_w + w] = grad_mask_val;
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
            const int n_kernels = out_w * weight_w * in_channels * batch_sz;
            const int n_mask_kernels = out_w * weight_w * n_mask_grps * batch_sz;
            const int mask_channels = weight_w * n_mask_grps;
            const int c_per_offset_grp = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_grp = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv1d_compute_grad_mask_cuda", ([&] {
                TVDCN_DISPATCH_CONDITION(deformable, ([&] {
                    deform_conv1d_compute_grad_mask_kernel<deformable><<<blocks, threads>>>(
                            n_kernels,
                            n_mask_kernels,
                            mask_channels,
                            c_per_offset_grp,
                            c_per_mask_grp,
                            columns.data_ptr<scalar_t>(),
                            input.data_ptr<scalar_t>(),
                            offset.data_ptr<scalar_t>(),
                            in_channels,
                            width,
                            weight_w,
                            pad_w,
                            stride_w,
                            dilation_w,
                            out_w,
                            batch_sz,
                            n_offset_grps,
                            n_mask_grps,
                            grad_mask.data_ptr<scalar_t>());
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in deform_conv1d_compute_grad_mask_cuda: %s\n", cudaGetErrorString(err));
            }
        }
    }
}
