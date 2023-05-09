#include <ATen/ATen.h>
#include "cuda_helpers.h"
#include "../utils/dispatch.h"

namespace tvdcn {
    namespace ops {
        namespace {
            constexpr float threadsFraction = 0.5;

            template<typename scalar_t>
            static __device__ __forceinline__ scalar_t sample(
                    const scalar_t *input,
                    const int depth,
                    const int height,
                    const int width,
                    const int z,
                    const int y,
                    const int x) {
                return (0 <= z && z < depth && 0 <= y && y < height && 0 <= x && x < width) ?
                       input[z * (width * height) + y * width + x] : static_cast<scalar_t>(0);
            }

            template<typename scalar_t>
            static __device__ scalar_t interpolate_sample(
                    const scalar_t *input,
                    const int depth,
                    const int height,
                    const int width,
                    const scalar_t z,
                    const scalar_t y,
                    const scalar_t x) {
                if (z <= -1 || depth <= z || y <= -1 || height <= y || x <= -1 || width <= x)
                    return 0;

                int z_l = floor(z);
                int z_h = z_l + 1;
                int y_l = floor(y);
                int y_h = y_l + 1;
                int x_l = floor(x);
                int x_h = x_l + 1;

                scalar_t dz_h = z - z_l;
                scalar_t dy_h = y - y_l;
                scalar_t dx_h = x - x_l;
                scalar_t dz_l = 1 - dz_h;
                scalar_t dy_l = 1 - dy_h;
                scalar_t dx_l = 1 - dx_h;

                bool valid_z_l = z_l >= 0;
                bool valid_z_h = z_h < depth;
                bool valid_y_l = y_l >= 0;
                bool valid_y_h = y_h < height;
                bool valid_x_l = x_l >= 0;
                bool valid_x_h = x_h < width;

                scalar_t val = 0;
                if (valid_z_l && valid_y_l && valid_x_l)
                    val += dz_l * dy_l * dx_l * input[z_l * (width * height) + y_l * width + x_l];
                if (valid_z_l && valid_y_l && valid_x_h)
                    val += dz_l * dy_l * dx_h * input[z_l * (width * height) + y_l * width + x_h];
                if (valid_z_l && valid_y_h && valid_x_l)
                    val += dz_l * dy_h * dx_l * input[z_l * (width * height) + y_h * width + x_l];
                if (valid_z_l && valid_y_h && valid_x_h)
                    val += dz_l * dy_h * dx_h * input[z_l * (width * height) + y_h * width + x_h];
                if (valid_z_h && valid_y_l && valid_x_l)
                    val += dz_h * dy_l * dx_l * input[z_h * (width * height) + y_l * width + x_l];
                if (valid_z_h && valid_y_l && valid_x_h)
                    val += dz_h * dy_l * dx_h * input[z_h * (width * height) + y_l * width + x_h];
                if (valid_z_h && valid_y_h && valid_x_l)
                    val += dz_h * dy_h * dx_l * input[z_h * (width * height) + y_h * width + x_l];
                if (valid_z_h && valid_y_h && valid_x_h)
                    val += dz_h * dy_h * dx_h * input[z_h * (width * height) + y_h * width + x_h];
                return val;
            }

            template<typename scalar_t>
            static __device__ __forceinline__ void insert(
                    scalar_t *output,
                    const int depth,
                    const int height,
                    const int width,
                    const int z,
                    const int y,
                    const int x,
                    const scalar_t val) {
                if (0 <= z && z < depth && 0 <= y && y < height && 0 <= x && x < width)
                    atomicAdd(output + z * height * width + y * width + x, val);
            }

            template<typename scalar_t>
            static __device__ void interpolate_insert(
                    scalar_t *output,
                    const int depth,
                    const int height,
                    const int width,
                    const scalar_t z,
                    const scalar_t y,
                    const scalar_t x,
                    const scalar_t val) {
                int z_l = floor(z);
                int z_h = z_l + 1;
                int y_l = floor(y);
                int y_h = y_l + 1;
                int x_l = floor(x);
                int x_h = x_l + 1;

                scalar_t dz_h = z - z_l;
                scalar_t dy_h = y - y_l;
                scalar_t dx_h = x - x_l;
                scalar_t dz_l = 1 - dz_h;
                scalar_t dy_l = 1 - dy_h;
                scalar_t dx_l = 1 - dx_h;

                bool valid_z_l = 0 <= z_l && z_l < depth;
                bool valid_z_h = 0 <= z_h && z_h < depth;
                bool valid_y_l = 0 <= y_l && y_l < height;
                bool valid_y_h = 0 <= y_h && y_h < height;
                bool valid_x_l = 0 <= x_l && x_l < width;
                bool valid_x_h = 0 <= x_h && x_h < width;

                if (valid_z_l && valid_y_l && valid_x_l)
                    atomicAdd(output + z_l * height * width + y_l * width + x_l, dz_l * dy_l * dx_l * val);
                if (valid_z_l && valid_y_l && valid_x_h)
                    atomicAdd(output + z_l * height * width + y_l * width + x_h, dz_l * dy_l * dx_h * val);
                if (valid_z_l && valid_y_h && valid_x_l)
                    atomicAdd(output + z_l * height * width + y_h * width + x_l, dz_l * dy_h * dx_l * val);
                if (valid_z_l && valid_y_h && valid_x_h)
                    atomicAdd(output + z_l * height * width + y_h * width + x_h, dz_l * dy_h * dx_h * val);
                if (valid_z_h && valid_y_l && valid_x_l)
                    atomicAdd(output + z_h * height * width + y_l * width + x_l, dz_h * dy_l * dx_l * val);
                if (valid_z_h && valid_y_l && valid_x_h)
                    atomicAdd(output + z_h * height * width + y_l * width + x_h, dz_h * dy_l * dx_h * val);
                if (valid_z_h && valid_y_h && valid_x_l)
                    atomicAdd(output + z_h * height * width + y_h * width + x_l, dz_h * dy_h * dx_l * val);
                if (valid_z_h && valid_y_h && valid_x_h)
                    atomicAdd(output + z_h * height * width + y_h * width + x_h, dz_h * dy_h * dx_h * val);
            }

            template<typename scalar_t>
            static __device__ scalar_t trilinear_coordinate_weight(
                    const scalar_t *input,
                    const int depth,
                    const int height,
                    const int width,
                    const scalar_t z,
                    const scalar_t y,
                    const scalar_t x,
                    const int direction) {
                int z_l = floor(z);
                int z_h = z_l + 1;
                int y_l = floor(y);
                int y_h = y_l + 1;
                int x_l = floor(x);
                int x_h = x_l + 1;

                scalar_t dz_h = (direction == 0) ? static_cast<scalar_t>(1) : z - z_l;
                scalar_t dy_h = (direction == 1) ? static_cast<scalar_t>(1) : y - y_l;
                scalar_t dx_h = (direction == 2) ? static_cast<scalar_t>(1) : x - x_l;
                scalar_t dz_l = (direction == 0) ? static_cast<scalar_t>(-1) : 1 - dz_h;
                scalar_t dy_l = (direction == 1) ? static_cast<scalar_t>(-1) : 1 - dy_h;
                scalar_t dx_l = (direction == 2) ? static_cast<scalar_t>(-1) : 1 - dx_h;

                bool valid_z_l = 0 <= z_l && z_l < depth;
                bool valid_z_h = 0 <= z_h && z_h < depth;
                bool valid_y_l = 0 <= y_l && y_l < height;
                bool valid_y_h = 0 <= y_h && y_h < height;
                bool valid_x_l = 0 <= x_l && x_l < width;
                bool valid_x_h = 0 <= x_h && x_h < width;

                scalar_t val = 0;
                if (valid_z_l && valid_y_l && valid_x_l)
                    val += dz_l * dy_l * dx_l * input[z_l * (width * height) + y_l * width + x_l];
                if (valid_z_l && valid_y_l && valid_x_h)
                    val += dz_l * dy_l * dx_h * input[z_l * (width * height) + y_l * width + x_h];
                if (valid_z_l && valid_y_h && valid_x_l)
                    val += dz_l * dy_h * dx_l * input[z_l * (width * height) + y_h * width + x_l];
                if (valid_z_l && valid_y_h && valid_x_h)
                    val += dz_l * dy_h * dx_h * input[z_l * (width * height) + y_h * width + x_h];
                if (valid_z_h && valid_y_l && valid_x_l)
                    val += dz_h * dy_l * dx_l * input[z_h * (width * height) + y_l * width + x_l];
                if (valid_z_h && valid_y_l && valid_x_h)
                    val += dz_h * dy_l * dx_h * input[z_h * (width * height) + y_l * width + x_h];
                if (valid_z_h && valid_y_h && valid_x_l)
                    val += dz_h * dy_h * dx_l * input[z_h * (width * height) + y_h * width + x_l];
                if (valid_z_h && valid_y_h && valid_x_h)
                    val += dz_h * dy_h * dx_h * input[z_h * (width * height) + y_h * width + x_h];
                return val;
            }
        }

        template<bool deformable, bool modulated, typename scalar_t>
        static __global__ void vol2col_kernel(
                const int n_kernels,
                const int c_per_offset_grp,
                const int c_per_mask_grp,
                const scalar_t *input,
                const scalar_t *offset,
                const scalar_t *mask,
                const int depth,
                const int height,
                const int width,
                const int weight_d,
                const int weight_h,
                const int weight_w,
                const int pad_d,
                const int pad_h,
                const int pad_w,
                const int stride_d,
                const int stride_h,
                const int stride_w,
                const int dilation_d,
                const int dilation_h,
                const int dilation_w,
                const int out_d,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int in_channels,
                const int n_offset_grps,
                const int n_mask_grps,
                scalar_t *columns) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                const int out_x = index % out_w;
                const int out_y = (index / out_w) % out_h;
                const int out_z = (index / (out_w * out_h)) % out_d;
                const int out_b = (index / (out_w * out_h * out_d)) % batch_sz;
                const int in_c = index / (out_w * out_h * out_d * batch_sz);
                const int out_c = in_c * weight_d * weight_h * weight_w;

                const int offset_grp = in_c / c_per_offset_grp;
                const int mask_grp = in_c / c_per_mask_grp;

                auto columns_ptr = columns +
                                   (out_c * (batch_sz * out_d * out_h * out_w) + out_b * (out_d * out_h * out_w) +
                                    out_z * (out_h * out_w) + out_y * out_w + out_x);
                auto input_ptr = input +
                                 (out_b * (in_channels * depth * height * width) + in_c * (depth * height * width));
                auto offset_ptr = offset +
                                  (out_b * n_offset_grps + offset_grp) * 3 * weight_d * weight_h * weight_w *
                                  out_d * out_h * out_w;
                auto mask_ptr = mask +
                                (out_b * n_mask_grps + mask_grp) * weight_d * weight_h * weight_w *
                                out_d * out_h * out_w;

                for (int i = 0; i < weight_d; ++i) {
                    for (int j = 0; j < weight_h; ++j) {
                        for (int k = 0; k < weight_w; ++k) {
                            const int mask_idx = i * weight_h * weight_w + j * weight_w + k;
                            const int offset_idx = 3 * mask_idx;

                            const scalar_t offset_d =
                                    deformable ?
                                    offset_ptr[((offset_idx * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                               : static_cast<scalar_t>(0);
                            const scalar_t offset_h =
                                    deformable ?
                                    offset_ptr[(((offset_idx + 1) * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                               : static_cast<scalar_t>(0);
                            const scalar_t offset_w =
                                    deformable ?
                                    offset_ptr[(((offset_idx + 2) * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                               : static_cast<scalar_t>(0);
                            const scalar_t z = (out_z * stride_d - pad_d) + i * dilation_d + offset_d;
                            const scalar_t y = (out_y * stride_h - pad_h) + j * dilation_h + offset_h;
                            const scalar_t x = (out_x * stride_w - pad_w) + k * dilation_w + offset_w;

                            const scalar_t val =
                                    deformable ?
                                    interpolate_sample(input_ptr, depth, height, width, z, y, x)
                                               : sample(input_ptr, depth, height, width,
                                                        static_cast<int>(z),
                                                        static_cast<int>(y),
                                                        static_cast<int>(x));

                            const scalar_t mask_val =
                                    modulated ?
                                    mask_ptr[((mask_idx * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                              : static_cast<scalar_t>(1);

                            *columns_ptr = val * mask_val;
                            columns_ptr += batch_sz * out_d * out_h * out_w;
                        }
                    }
                }
            }
        }

        void vol2col_cuda(
                const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int in_channels,
                const int depth,
                const int height,
                const int width,
                const int weight_d,
                const int weight_h,
                const int weight_w,
                const int pad_d,
                const int pad_h,
                const int pad_w,
                const int stride_d,
                const int stride_h,
                const int stride_w,
                const int dilation_d,
                const int dilation_h,
                const int dilation_w,
                const int out_d,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &columns) {
            const int n_kernels = in_channels * out_d * out_h * out_w * batch_sz;
            const int c_per_offset_grp = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_grp = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    input.scalar_type(), "vol2col_cuda", ([&] {
                TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                    vol2col_kernel<deformable, modulated><<<blocks, threads>>>(
                            n_kernels,
                            c_per_offset_grp,
                            c_per_mask_grp,
                            input.data_ptr<scalar_t>(),
                            offset.data_ptr<scalar_t>(),
                            mask.data_ptr<scalar_t>(),
                            depth,
                            height,
                            width,
                            weight_d,
                            weight_h,
                            weight_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            out_d,
                            out_h,
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
                printf("error in vol2col_cuda: %s\n", cudaGetErrorString(err));
            }
        }

        template<bool deformable, bool modulated, typename scalar_t>
        static __global__ void col2vol_kernel(
                const int n_kernels,
                const int c_per_offset_grp,
                const int c_per_mask_grp,
                const scalar_t *columns,
                const scalar_t *offset,
                const scalar_t *mask,
                const int in_channels,
                const int depth,
                const int height,
                const int width,
                const int weight_d,
                const int weight_h,
                const int weight_w,
                const int pad_d,
                const int pad_h,
                const int pad_w,
                const int stride_d,
                const int stride_h,
                const int stride_w,
                const int dilation_d,
                const int dilation_h,
                const int dilation_w,
                const int out_d,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                scalar_t *grad_input) {
            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                const int out_x = index % out_w;
                const int out_y = (index / out_w) % out_h;
                const int out_z = (index / (out_w * out_h)) % out_d;
                const int b = (index / (out_w * out_h * out_d)) % batch_sz;
                const int k = (index / (out_w * out_h * out_d * batch_sz)) % weight_w;
                const int j = (index / (out_w * out_h * out_d * batch_sz * weight_w)) % weight_h;
                const int i = (index / (out_w * out_h * out_d * batch_sz * weight_w * weight_h)) % weight_d;
                const int c = index / (out_w * out_h * out_d * batch_sz * weight_w * weight_h * weight_d);

                const int offset_grp = c / c_per_offset_grp;
                const int mask_grp = c / c_per_mask_grp;

                const int mask_idx = (i * weight_h * weight_w + j * weight_w + k);
                const int offset_idx = 3 * mask_idx;

                auto offset_ptr = offset +
                                  (b * n_offset_grps + offset_grp) * 3 * weight_d * weight_h * weight_w *
                                  out_d * out_h * out_w;
                auto mask_ptr = mask +
                                (b * n_mask_grps + mask_grp) * weight_d * weight_h * weight_w * out_d * out_h * out_w;

                const scalar_t offset_d =
                        deformable ?
                        offset_ptr[((offset_idx * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                   : static_cast<scalar_t>(0);
                const scalar_t offset_h =
                        deformable ?
                        offset_ptr[(((offset_idx + 1) * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                   : static_cast<scalar_t>(0);
                const scalar_t offset_w =
                        deformable ?
                        offset_ptr[(((offset_idx + 2) * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                   : static_cast<scalar_t>(0);
                const scalar_t z = (out_z * stride_d - pad_d) + i * dilation_d + offset_d;
                const scalar_t y = (out_y * stride_h - pad_h) + j * dilation_h + offset_h;
                const scalar_t x = (out_x * stride_w - pad_w) + k * dilation_w + offset_w;

                const scalar_t mask_val =
                        modulated ?
                        mask_ptr[((mask_idx * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                  : static_cast<scalar_t>(1);

                const scalar_t val = columns[index] * mask_val;

                auto grad_input_ptr = grad_input +
                                      (b * in_channels + c) * depth * height * width;
                if (deformable)
                    interpolate_insert(grad_input_ptr, depth, height, width, z, y, x, val);
                else
                    insert(grad_input_ptr, depth, height, width, z, y, x, val);
            }
        }

        void col2vol_cuda(
                const at::Tensor &columns,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int in_channels,
                const int depth,
                const int height,
                const int width,
                const int weight_d,
                const int weight_h,
                const int weight_w,
                const int pad_d,
                const int pad_h,
                const int pad_w,
                const int stride_d,
                const int stride_h,
                const int stride_w,
                const int dilation_d,
                const int dilation_h,
                const int dilation_w,
                const int out_d,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_input) {
            const int n_kernels = in_channels * weight_d * weight_h * weight_w * out_d * out_h * out_w * batch_sz;
            const int c_per_offset_grp = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_grp = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "col2vol_cuda", ([&] {
                TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                    col2vol_kernel<deformable, modulated><<<blocks, threads>>>(
                            n_kernels,
                            c_per_offset_grp,
                            c_per_mask_grp,
                            columns.data_ptr<scalar_t>(),
                            offset.data_ptr<scalar_t>(),
                            mask.data_ptr<scalar_t>(),
                            in_channels,
                            depth,
                            height,
                            width,
                            weight_d,
                            weight_h,
                            weight_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            out_d,
                            out_h,
                            out_w,
                            batch_sz,
                            n_offset_grps,
                            n_mask_grps,
                            grad_input.data_ptr<scalar_t>());
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in col2vol_cuda: %s\n", cudaGetErrorString(err));
            }
        }

        template<bool modulated, typename scalar_t>
        static __global__ void deform_conv3d_compute_grad_offset_kernel(
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
                const int depth,
                const int height,
                const int width,
                const int weight_d,
                const int weight_h,
                const int weight_w,
                const int pad_d,
                const int pad_h,
                const int pad_w,
                const int stride_d,
                const int stride_h,
                const int stride_w,
                const int dilation_d,
                const int dilation_h,
                const int dilation_w,
                const int out_d,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                scalar_t *grad_offset) {
            CUDA_1D_KERNEL_LOOP(index, n_offset_kernels) {
                scalar_t grad_offset_val = 0;

                const int w = index % out_w;
                const int h = (index / out_w) % out_h;
                const int d = (index / (out_w * out_h)) % out_d;
                const int c = (index / (out_w * out_h * out_d)) % offset_channels;
                const int b = index / (out_w * out_h * out_d * offset_channels);

                const int offset_grp = c / (3 * weight_d * weight_h * weight_w);

                const int col_offset = offset_grp * c_per_offset_grp * weight_d * weight_h * weight_w * batch_sz *
                                       out_d * out_h * out_w;
                auto columns_ptr = columns + col_offset;
                auto input_ptr = input +
                                 (b * n_offset_grps + offset_grp) * c_per_offset_grp * depth * height * width;
                auto offset_ptr = offset +
                                  (b * n_offset_grps + offset_grp) * 3 * weight_d * weight_h * weight_w *
                                  out_d * out_h * out_w;

                const int offset_c = c - offset_grp * 3 * weight_d * weight_h * weight_w;
                const int direction = offset_c % 3;

                const int c_bound = c_per_offset_grp * weight_d * weight_h * weight_w;
                const int col_step = weight_d * weight_h * weight_w;
                for (int col_c = (offset_c / 3); col_c < c_bound; col_c += col_step) {
                    const int col_pos = (((col_c * batch_sz + b) * out_d + d) * out_h + h) * out_w + w;
                    const int in_c = (col_offset + col_pos) * in_channels / n_kernels;

                    const int mask_grp = in_c / c_per_mask_grp;
                    auto mask_ptr = mask +
                                    (b * n_mask_grps + mask_grp) * weight_d * weight_h * weight_w *
                                    out_d * out_h * out_w;

                    const int out_x = col_pos % out_w;
                    const int out_y = (col_pos / out_w) % out_h;
                    const int out_z = (col_pos / (out_w * out_h)) % out_d;
                    const int k = (col_pos / (out_w * out_h * out_d * batch_sz)) % weight_w;
                    const int j = (col_pos / (out_w * out_h * out_d * batch_sz * weight_w)) % weight_h;
                    const int i = (col_pos / (out_w * out_h * out_d * batch_sz * weight_w * weight_h)) % weight_d;

                    const int mask_idx = i * weight_h * weight_w + j * weight_w + k;
                    const int offset_idx = 3 * mask_idx;

                    const scalar_t offset_d =
                            offset_ptr[((offset_idx * out_d + out_z) * out_h + out_y) * out_w + out_x];
                    const scalar_t offset_h =
                            offset_ptr[(((offset_idx + 1) * out_d + out_z) * out_h + out_y) * out_w + out_x];
                    const scalar_t offset_w =
                            offset_ptr[(((offset_idx + 2) * out_d + out_z) * out_h + out_y) * out_w + out_x];

                    const scalar_t z = (out_z * stride_d - pad_d) + i * dilation_d + offset_d;
                    const scalar_t y = (out_y * stride_h - pad_h) + j * dilation_h + offset_h;
                    const scalar_t x = (out_x * stride_w - pad_w) + k * dilation_w + offset_w;

                    const scalar_t mask_val =
                            modulated ?
                            mask_ptr[((mask_idx * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                      : static_cast<scalar_t>(1);

                    const scalar_t weight =
                            trilinear_coordinate_weight(input_ptr, depth, height, width, z, y, x, direction);

                    grad_offset_val += columns_ptr[col_pos] * weight * mask_val;
                    input_ptr += depth * height * width;
                }

                grad_offset[index] = grad_offset_val;
            }
        }

        void deform_conv3d_compute_grad_offset_cuda(
                const at::Tensor &columns,
                const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const int in_channels,
                const int depth,
                const int height,
                const int width,
                const int weight_d,
                const int weight_h,
                const int weight_w,
                const int pad_d,
                const int pad_h,
                const int pad_w,
                const int stride_d,
                const int stride_h,
                const int stride_w,
                const int dilation_d,
                const int dilation_h,
                const int dilation_w,
                const int out_d,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_offset) {
            if (!deformable) return;
            const int n_kernels = out_d * out_h * out_w * weight_d * weight_h * weight_w * in_channels * batch_sz;
            const int n_offset_kernels =
                    out_d * out_h * out_w * 3 * weight_d * weight_h * weight_w * n_offset_grps * batch_sz;
            const int offset_channels = 3 * weight_d * weight_h * weight_w * n_offset_grps;
            const int c_per_offset_grp = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_grp = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv3d_compute_grad_offset_cuda", ([&] {
                TVDCN_DISPATCH_CONDITION(modulated, ([&] {
                    deform_conv3d_compute_grad_offset_kernel<modulated><<<blocks, threads>>>(
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
                            depth,
                            height,
                            width,
                            weight_d,
                            weight_h,
                            weight_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            out_d,
                            out_h,
                            out_w,
                            batch_sz,
                            n_offset_grps,
                            n_mask_grps,
                            grad_offset.data_ptr<scalar_t>());
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in deform_conv3d_compute_grad_offset_cuda: %s\n", cudaGetErrorString(err));
            }
        }

        template<bool deformable, typename scalar_t>
        static __global__ void deform_conv3d_compute_grad_mask_kernel(
                const int n_kernels,
                const int n_mask_kernels,
                const int mask_channels,
                const int c_per_offset_grp,
                const int c_per_mask_grp,
                const scalar_t *columns,
                const scalar_t *input,
                const scalar_t *offset,
                const int in_channels,
                const int depth,
                const int height,
                const int width,
                const int weight_d,
                const int weight_h,
                const int weight_w,
                const int pad_d,
                const int pad_h,
                const int pad_w,
                const int stride_d,
                const int stride_h,
                const int stride_w,
                const int dilation_d,
                const int dilation_h,
                const int dilation_w,
                const int out_d,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                scalar_t *grad_mask) {
            CUDA_1D_KERNEL_LOOP(index, n_mask_kernels) {
                scalar_t grad_mask_val = 0;

                const int w = index % out_w;
                const int h = (index / out_w) % out_h;
                const int d = (index / (out_w * out_h)) % out_d;
                const int c = (index / (out_w * out_h * out_d)) % mask_channels;
                const int b = index / (out_w * out_h * out_d * mask_channels);

                const int mask_grp = c / (weight_d * weight_h * weight_w);

                const int col_offset = mask_grp * c_per_mask_grp * weight_d * weight_h * weight_w * batch_sz *
                                       out_d * out_h * out_w;
                auto columns_ptr = columns + col_offset;
                auto input_ptr = input +
                                 (b * n_mask_grps + mask_grp) * c_per_mask_grp * depth * height * width;
                auto grad_mask_ptr = grad_mask +
                                (b * n_mask_grps + mask_grp) * weight_d * weight_h * weight_w * out_d * out_h * out_w;

                const int mask_c = c - mask_grp * weight_d * weight_h * weight_w;

                const int c_bound = c_per_mask_grp * weight_d * weight_h * weight_w;
                const int col_step = weight_d * weight_h * weight_w;
                for (int col_c = mask_c; col_c < c_bound; col_c += col_step) {
                    const int col_pos = (((col_c * batch_sz + b) * out_d + d) * out_h + h) * out_w + w;
                    const int in_c = (col_offset + col_pos) * in_channels / n_kernels;

                    const int offset_grp = in_c / c_per_offset_grp;
                    auto offset_ptr = offset +
                                      (b * n_offset_grps + offset_grp) * 3 * weight_d * weight_h * weight_w *
                                      out_d * out_h * out_w;

                    const int out_x = col_pos % out_w;
                    const int out_y = (col_pos / out_w) % out_h;
                    const int out_z = (col_pos / (out_w * out_h)) % out_d;
                    const int k = (col_pos / (out_w * out_h * out_d * batch_sz)) % weight_w;
                    const int j = (col_pos / (out_w * out_h * out_d * batch_sz * weight_w)) % weight_h;
                    const int i = (col_pos / (out_w * out_h * out_d * batch_sz * weight_w * weight_h)) % weight_d;

                    const int mask_idx = i * weight_h * weight_w + j * weight_w + k;
                    const int offset_idx = 3 * mask_idx;

                    const scalar_t offset_d =
                            deformable ?
                            offset_ptr[((offset_idx * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                       : static_cast<scalar_t>(0);
                    const scalar_t offset_h =
                            deformable ?
                            offset_ptr[(((offset_idx + 1) * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                       : static_cast<scalar_t>(0);
                    const scalar_t offset_w =
                            deformable ?
                            offset_ptr[(((offset_idx + 2) * out_d + out_z) * out_h + out_y) * out_w + out_x]
                                       : static_cast<scalar_t>(0);

                    const scalar_t z = (out_z * stride_d - pad_d) + i * dilation_d + offset_d;
                    const scalar_t y = (out_y * stride_h - pad_h) + j * dilation_h + offset_h;
                    const scalar_t x = (out_x * stride_w - pad_w) + k * dilation_w + offset_w;

                    const scalar_t val =
                            deformable ?
                            interpolate_sample(input_ptr, depth, height, width, z, y, x)
                                       : sample(input_ptr, depth, height, width,
                                                static_cast<int>(z),
                                                static_cast<int>(y),
                                                static_cast<int>(x));

                    grad_mask_val += columns_ptr[col_pos] * val;
                    input_ptr += depth * height * width;
                }

                grad_mask_ptr[((mask_c * out_d + d) * out_h + h) * out_w + w] = grad_mask_val;
            }
        }

        void deform_conv3d_compute_grad_mask_cuda(
                const at::Tensor &columns,
                const at::Tensor &input,
                const at::Tensor &offset,
                const int in_channels,
                const int depth,
                const int height,
                const int width,
                const int weight_d,
                const int weight_h,
                const int weight_w,
                const int pad_d,
                const int pad_h,
                const int pad_w,
                const int stride_d,
                const int stride_h,
                const int stride_w,
                const int dilation_d,
                const int dilation_h,
                const int dilation_w,
                const int out_d,
                const int out_h,
                const int out_w,
                const int batch_sz,
                const int n_offset_grps,
                const int n_mask_grps,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_mask) {
            if (!modulated) return;
            const int n_kernels = out_d * out_h * out_w * weight_d * weight_h * weight_w * in_channels * batch_sz;
            const int n_mask_kernels =
                    out_d * out_h * out_w * weight_d * weight_h * weight_w * n_mask_grps * batch_sz;
            const int mask_channels = weight_d * weight_h * weight_w * n_mask_grps;
            const int c_per_offset_grp = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_grp = modulated ? in_channels / n_mask_grps : 1;

            const unsigned int threads = GET_THREADS(threadsFraction);
            const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv3d_compute_grad_mask_cuda", ([&] {
                TVDCN_DISPATCH_CONDITION(deformable, ([&] {
                    deform_conv3d_compute_grad_mask_kernel<deformable><<<blocks, threads>>>(
                            n_kernels,
                            n_mask_kernels,
                            mask_channels,
                            c_per_offset_grp,
                            c_per_mask_grp,
                            columns.data_ptr<scalar_t>(),
                            input.data_ptr<scalar_t>(),
                            offset.data_ptr<scalar_t>(),
                            in_channels,
                            depth,
                            height,
                            width,
                            weight_d,
                            weight_h,
                            weight_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            out_d,
                            out_h,
                            out_w,
                            batch_sz,
                            n_offset_grps,
                            n_mask_grps,
                            grad_mask.data_ptr<scalar_t>());
                }));
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error in deform_conv3d_compute_grad_mask_cuda: %s\n", cudaGetErrorString(err));
            }
        }
    }
}
