#include <ATen/ATen.h>
#include "cpu_helpers.h"
#include "../utils/dispatch.h"

namespace tvdcn {
    namespace ops {
        namespace {
            template<typename scalar_t>
            __forceinline__ scalar_t sample(
                    const at::TensorAccessor<scalar_t, 5> input,
                    const int b,
                    const int c,
                    const int depth,
                    const int height,
                    const int width,
                    const int z,
                    const int y,
                    const int x) {
                return (0 <= z && z < depth && 0 <= y && y < height && 0 <= x && x < width) ?
                       input[b][c][z][y][x] : static_cast<scalar_t>(0);
            }

            template<typename scalar_t>
            __forceinline__ scalar_t interpolate_sample(
                    const at::TensorAccessor<scalar_t, 5> input,
                    const int b,
                    const int c,
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
                    val += dz_l * dy_l * dx_l * input[b][c][z_l][y_l][x_l];
                if (valid_z_l && valid_y_l && valid_x_h)
                    val += dz_l * dy_l * dx_h * input[b][c][z_l][y_l][x_h];
                if (valid_z_l && valid_y_h && valid_x_l)
                    val += dz_l * dy_h * dx_l * input[b][c][z_l][y_h][x_l];
                if (valid_z_l && valid_y_h && valid_x_h)
                    val += dz_l * dy_h * dx_h * input[b][c][z_l][y_h][x_h];
                if (valid_z_h && valid_y_l && valid_x_l)
                    val += dz_h * dy_l * dx_l * input[b][c][z_h][y_l][x_l];
                if (valid_z_h && valid_y_l && valid_x_h)
                    val += dz_h * dy_l * dx_h * input[b][c][z_h][y_l][x_h];
                if (valid_z_h && valid_y_h && valid_x_l)
                    val += dz_h * dy_h * dx_l * input[b][c][z_h][y_h][x_l];
                if (valid_z_h && valid_y_h && valid_x_h)
                    val += dz_h * dy_h * dx_h * input[b][c][z_h][y_h][x_h];
                return val;
            }

            template<typename scalar_t>
            __forceinline__ void insert(
                    at::TensorAccessor<scalar_t, 5> output,
                    const int b,
                    const int c,
                    const int depth,
                    const int height,
                    const int width,
                    const int z,
                    const int y,
                    const int x,
                    const scalar_t val) {
                if (0 <= z && z < depth && 0 <= y && y < height && 0 <= x && x < width)
                    output[b][c][z][y][x] += val;
            }

            template<typename scalar_t>
            __forceinline__ void interpolate_insert(
                    at::TensorAccessor<scalar_t, 5> output,
                    const int b,
                    const int c,
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
                    output[b][c][z_l][y_l][x_l] += dz_l * dy_l * dx_l * val;
                if (valid_z_l && valid_y_l && valid_x_h)
                    output[b][c][z_l][y_l][x_h] += dz_l * dy_l * dx_h * val;
                if (valid_z_l && valid_y_h && valid_x_l)
                    output[b][c][z_l][y_h][x_l] += dz_l * dy_h * dx_l * val;
                if (valid_z_l && valid_y_h && valid_x_h)
                    output[b][c][z_l][y_h][x_h] += dz_l * dy_h * dx_h * val;
                if (valid_z_h && valid_y_l && valid_x_l)
                    output[b][c][z_h][y_l][x_l] += dz_h * dy_l * dx_l * val;
                if (valid_z_h && valid_y_l && valid_x_h)
                    output[b][c][z_h][y_l][x_h] += dz_h * dy_l * dx_h * val;
                if (valid_z_h && valid_y_h && valid_x_l)
                    output[b][c][z_h][y_h][x_l] += dz_h * dy_h * dx_l * val;
                if (valid_z_h && valid_y_h && valid_x_h)
                    output[b][c][z_h][y_h][x_h] += dz_h * dy_h * dx_h * val;
            }

            template<typename scalar_t>
            __forceinline__ scalar_t coordinate_weight(
                    const at::TensorAccessor<scalar_t, 5> input,
                    const int b,
                    const int c,
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
                    val += dz_l * dy_l * dx_l * input[b][c][z_l][y_l][x_l];
                if (valid_z_l && valid_y_l && valid_x_h)
                    val += dz_l * dy_l * dx_h * input[b][c][z_l][y_l][x_h];
                if (valid_z_l && valid_y_h && valid_x_l)
                    val += dz_l * dy_h * dx_l * input[b][c][z_l][y_h][x_l];
                if (valid_z_l && valid_y_h && valid_x_h)
                    val += dz_l * dy_h * dx_h * input[b][c][z_l][y_h][x_h];
                if (valid_z_h && valid_y_l && valid_x_l)
                    val += dz_h * dy_l * dx_l * input[b][c][z_h][y_l][x_l];
                if (valid_z_h && valid_y_l && valid_x_h)
                    val += dz_h * dy_l * dx_h * input[b][c][z_h][y_l][x_h];
                if (valid_z_h && valid_y_h && valid_x_l)
                    val += dz_h * dy_h * dx_l * input[b][c][z_h][y_h][x_l];
                if (valid_z_h && valid_y_h && valid_x_h)
                    val += dz_h * dy_h * dx_h * input[b][c][z_h][y_h][x_h];
                return val;
            }
        }

        template<bool deformable, bool modulated, typename scalar_t>
        static void vol2col_kernel(
                const int n_kernels,
                const at::TensorAccessor<scalar_t, 5> input,
                const at::TensorAccessor<scalar_t, 9> offset,
                const at::TensorAccessor<scalar_t, 8> mask,
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
                const int in_channels,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::TensorAccessor<scalar_t, 8> columns) {
            CPU_1D_KERNEL_LOOP(index, n_kernels) {
                const int w = index % out_w;
                const int h = (index / out_w) % out_h;
                const int d = (index / (out_w * out_h)) % out_d;
                const int c = (index / (out_w * out_h * out_d)) % in_channels;
                const int b = index / (out_w * out_h * out_d * in_channels);

                const int offset_group_idx = c / c_per_offset_group;
                const int mask_group_idx = c / c_per_mask_group;

                for (int i = 0; i < weight_d; ++i) {
                    for (int j = 0; j < weight_h; ++j) {
                        for (int k = 0; k < weight_w; ++k) {
                            const int z = (d * stride_d - pad_d) + i * dilation_d;
                            const int y = (h * stride_h - pad_h) + j * dilation_h;
                            const int x = (w * stride_w - pad_w) + k * dilation_w;

                            const scalar_t val =
                                    deformable ?
                                    interpolate_sample(
                                            input, b, c, depth, height, width,
                                            z + offset[b][offset_group_idx][i][j][k][0][d][h][w],
                                            y + offset[b][offset_group_idx][i][j][k][1][d][h][w],
                                            x + offset[b][offset_group_idx][i][j][k][2][d][h][w])
                                               : sample(input, b, c, depth, height, width, z, y, x);

                            const scalar_t mask_val =
                                    modulated ?
                                    mask[b][mask_group_idx][i][j][k][d][h][w] : static_cast<scalar_t>(1);

                            columns[c][i][j][k][b][d][h][w] = val * mask_val;
                        }
                    }
                }
            }
        }

        void vol2col_cpu(
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
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    input.scalar_type(), "vol2col_cpu", ([&] {
                auto columns_accessor = columns.accessor<scalar_t, 8>();
                TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                    vol2col_kernel<deformable, modulated>(
                            n_kernels,
                            input.accessor<scalar_t, 5>(),
                            offset.accessor<scalar_t, 9>(),
                            mask.accessor<scalar_t, 8>(),
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
                            in_channels,
                            c_per_offset_group,
                            c_per_mask_group,
                            columns_accessor);
                }));
            }));
        }

        template<bool deformable, bool modulated, typename scalar_t>
        static void col2vol_kernel(
                const int n_kernels,
                const at::TensorAccessor<scalar_t, 8> columns,
                const at::TensorAccessor<scalar_t, 9> offset,
                const at::TensorAccessor<scalar_t, 8> mask,
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
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::TensorAccessor<scalar_t, 5> grad_input) {
            CPU_1D_KERNEL_LOOP(index, n_kernels) {
                const int k = index % weight_w;
                const int j = (index / weight_w) % weight_h;
                const int i = (index / (weight_w * weight_h)) % weight_d;
                const int w = (index / (weight_w * weight_h * weight_d)) % out_w;
                const int h = (index / (weight_w * weight_h * weight_d * out_w)) % out_h;
                const int d = (index / (weight_w * weight_h * weight_d * out_w * out_h)) % out_d;
                const int c = (index / (weight_w * weight_h * weight_d * out_w * out_h * out_d)) % in_channels;
                const int b = (index / (weight_w * weight_h * weight_d * out_w * out_h * out_d * in_channels));

                const int offset_group_idx = c / c_per_offset_group;
                const int mask_group_idx = c / c_per_mask_group;

                const int z = (d * stride_d - pad_d) + i * dilation_d;
                const int y = (h * stride_h - pad_h) + j * dilation_h;
                const int x = (w * stride_w - pad_w) + k * dilation_w;

                const scalar_t mask_val =
                        modulated ?
                        mask[b][mask_group_idx][i][j][k][d][h][w] : static_cast<scalar_t>(1);

                const scalar_t val = columns[c][i][j][k][b][d][h][w] * mask_val;

                if (deformable)
                    interpolate_insert(
                            grad_input, b, c, depth, height, width,
                            z + offset[b][offset_group_idx][i][j][k][0][d][h][w],
                            y + offset[b][offset_group_idx][i][j][k][1][d][h][w],
                            x + offset[b][offset_group_idx][i][j][k][2][d][h][w],
                            val);
                else
                    insert(grad_input, b, c, depth, height, width, z, y, x, val);
            }
        }

        void col2vol_cpu(
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
            const int n_kernels = batch_sz * in_channels * out_d * out_h * out_w * weight_d * weight_h * weight_w;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "col2vol_cpu", ([&] {
                auto grad_input_accessor = grad_input.accessor<scalar_t, 5>();
                TVDCN_DISPATCH_CONDITION2(deformable, modulated, ([&] {
                    col2vol_kernel<deformable, modulated>(
                            n_kernels,
                            columns.accessor<scalar_t, 8>(),
                            offset.accessor<scalar_t, 9>(),
                            mask.accessor<scalar_t, 8>(),
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
                            c_per_offset_group,
                            c_per_mask_group,
                            grad_input_accessor);
                }));
            }));
        }

        template<bool modulated, typename scalar_t>
        static void deform_conv3d_compute_grad_offset_kernel(
                const int n_kernels,
                const at::TensorAccessor<scalar_t, 8> columns,
                const at::TensorAccessor<scalar_t, 5> input,
                const at::TensorAccessor<scalar_t, 9> offset,
                const at::TensorAccessor<scalar_t, 8> mask,
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
                const int n_offset_grps,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::TensorAccessor<scalar_t, 9> grad_offset) {
            CPU_1D_KERNEL_LOOP(index, n_kernels) {
                const int o = index % 3;
                const int k = (index / 3) % weight_w;
                const int j = (index / (3 * weight_w)) % weight_h;
                const int i = (index / (3 * weight_w * weight_h)) % weight_d;
                const int w = (index / (3 * weight_w * weight_h * weight_d)) % out_w;
                const int h = (index / (3 * weight_w * weight_h * weight_d * out_w)) % out_h;
                const int d = (index / (3 * weight_w * weight_h * weight_d * out_w * out_h)) % out_d;
                const int g = (index / (3 * weight_w * weight_h * weight_d * out_w * out_h * out_d)) % n_offset_grps;
                const int b = index / (3 * weight_w * weight_h * weight_d * out_w * out_h * out_d * n_offset_grps);

                scalar_t grad_offset_val = 0;

                const int c_start = g * c_per_offset_group;
                const int c_end = c_start + c_per_offset_group;
                for (int c = c_start; c < c_end; ++c) {
                    const int mask_group_idx = c / c_per_mask_group;

                    const int z = (d * stride_d - pad_d) + i * dilation_d;
                    const int y = (h * stride_h - pad_h) + j * dilation_h;
                    const int x = (w * stride_w - pad_w) + k * dilation_w;

                    const scalar_t weight =
                            coordinate_weight(
                                    input, b, c, depth, height, width,
                                    z + offset[b][g][i][j][k][0][d][h][w],
                                    y + offset[b][g][i][j][k][1][d][h][w],
                                    x + offset[b][g][i][j][k][2][d][h][w],
                                    o);

                    const scalar_t mask_val =
                            modulated ?
                            mask[b][mask_group_idx][i][j][k][d][h][w] : static_cast<scalar_t>(1);

                    grad_offset_val += columns[c][i][j][k][b][d][h][w] * weight * mask_val;
                }

                grad_offset[b][g][i][j][k][o][d][h][w] = grad_offset_val;
            }
        }

        void deform_conv3d_compute_grad_offset_cpu(
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
            const int n_kernels = batch_sz * n_offset_grps * out_d * out_h * out_w * weight_d * weight_h * weight_w * 3;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv3d_compute_grad_offset_cpu", ([&] {
                auto grad_offset_accessor = grad_offset.accessor<scalar_t, 9>();
                TVDCN_DISPATCH_CONDITION(modulated, ([&] {
                    deform_conv3d_compute_grad_offset_kernel<modulated>(
                            n_kernels,
                            columns.accessor<scalar_t, 8>(),
                            input.accessor<scalar_t, 5>(),
                            offset.accessor<scalar_t, 9>(),
                            mask.accessor<scalar_t, 8>(),
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
                            n_offset_grps,
                            c_per_offset_group,
                            c_per_mask_group,
                            grad_offset_accessor);
                }));
            }));
        }

        template<bool deformable, typename scalar_t>
        static void deform_conv3d_compute_grad_mask_kernel(
                const int n_kernels,
                const at::TensorAccessor<scalar_t, 8> columns,
                const at::TensorAccessor<scalar_t, 5> input,
                const at::TensorAccessor<scalar_t, 9> offset,
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
                const int n_mask_grps,
                const int c_per_offset_group,
                const int c_per_mask_group,
                at::TensorAccessor<scalar_t, 8> grad_mask) {
            CPU_1D_KERNEL_LOOP(index, n_kernels) {
                const int k = index % weight_w;
                const int j = (index / weight_w) % weight_h;
                const int i = (index / (weight_w * weight_h)) % weight_d;
                const int w = (index / (weight_w * weight_h * weight_d)) % out_w;
                const int h = (index / (weight_w * weight_h * weight_d * out_w)) % out_h;
                const int d = (index / (weight_w * weight_h * weight_d * out_w * out_h)) % out_d;
                const int g = (index / (weight_w * weight_h * weight_d * out_w * out_h * out_d)) % n_mask_grps;
                const int b = index / (weight_w * weight_h * weight_d * out_w * out_h * out_d * n_mask_grps);

                scalar_t grad_mask_val = 0;

                const int c_start = g * c_per_mask_group;
                const int c_end = c_start + c_per_mask_group;
                for (int c = c_start; c < c_end; ++c) {
                    const int offset_group_idx = c / c_per_offset_group;

                    const int z = (d * stride_d - pad_d) + i * dilation_d;
                    const int y = (h * stride_h - pad_h) + j * dilation_h;
                    const int x = (w * stride_w - pad_w) + k * dilation_w;

                    const scalar_t val =
                            deformable ?
                            interpolate_sample(
                                    input, b, c, depth, height, width,
                                    z + offset[b][offset_group_idx][i][j][k][0][d][h][w],
                                    y + offset[b][offset_group_idx][i][j][k][1][d][h][w],
                                    x + offset[b][offset_group_idx][i][j][k][2][d][h][w])
                                       : sample(input, b, c, depth, height, width, z, y, x);

                    grad_mask_val += columns[c][i][j][k][b][d][h][w] * val;
                }

                grad_mask[b][g][i][j][k][d][h][w] = grad_mask_val;
            }
        }

        void deform_conv3d_compute_grad_mask_cpu(
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
            const int n_kernels = batch_sz * n_mask_grps * out_d * out_h * out_w * weight_d * weight_h * weight_w;
            const int c_per_offset_group = deformable ? in_channels / n_offset_grps : 1;
            const int c_per_mask_group = modulated ? in_channels / n_mask_grps : 1;

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    columns.scalar_type(), "deform_conv3d_compute_grad_mask_cpu", ([&] {
                auto grad_mask_accessor = grad_mask.accessor<scalar_t, 8>();
                TVDCN_DISPATCH_CONDITION(deformable, ([&] {
                    deform_conv3d_compute_grad_mask_kernel<deformable>(
                            n_kernels,
                            columns.accessor<scalar_t, 8>(),
                            input.accessor<scalar_t, 5>(),
                            offset.accessor<scalar_t, 9>(),
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
                            n_mask_grps,
                            c_per_offset_group,
                            c_per_mask_group,
                            grad_mask_accessor);
                }));
            }));
        }
    }
}
