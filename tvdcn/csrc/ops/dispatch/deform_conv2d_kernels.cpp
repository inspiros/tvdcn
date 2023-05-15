#include "../cpu/deform_conv2d_kernels_cpu.h"

#ifdef WITH_CUDA

#include "../cuda/deform_conv2d_kernels_cuda.h"

#endif

namespace tvdcn {
    namespace ops {
        void im2col(
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
                const int offset_groups,
                const int mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &columns) {
            if (input.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
                im2col_cuda(input,
                            offset,
                            mask,
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
                            batch_sz,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            columns);
#else
                AT_ERROR("Not compiled with GPU support");
#endif
            } else {
                im2col_cpu(input,
                           offset,
                           mask,
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
                           batch_sz,
                           offset_groups,
                           mask_groups,
                           deformable,
                           modulated,
                           columns);
            }
        }

        void col2im(
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
                const int offset_groups,
                const int mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_input) {
            if (grad_input.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
                col2im_cuda(columns,
                            offset,
                            mask,
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
                            batch_sz,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            grad_input);
#else
                AT_ERROR("Not compiled with GPU support");
#endif
            } else {
                col2im_cpu(columns,
                           offset,
                           mask,
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
                           batch_sz,
                           offset_groups,
                           mask_groups,
                           deformable,
                           modulated,
                           grad_input);
            }
        }

        void deform_conv2d_compute_grad_offset(
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
                const int offset_groups,
                const int mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_offset) {
            if (input.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
                deform_conv2d_compute_grad_offset_cuda(columns,
                                                       input,
                                                       offset,
                                                       mask,
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
                                                       batch_sz,
                                                       offset_groups,
                                                       mask_groups,
                                                       deformable,
                                                       modulated,
                                                       grad_offset);
#else
                AT_ERROR("Not compiled with GPU support");
#endif
            } else {
                deform_conv2d_compute_grad_offset_cpu(columns,
                                                      input,
                                                      offset,
                                                      mask,
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
                                                      batch_sz,
                                                      offset_groups,
                                                      mask_groups,
                                                      deformable,
                                                      modulated,
                                                      grad_offset);
            }
        }

        void deform_conv2d_compute_grad_mask(
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
                const int offset_groups,
                const int mask_groups,
                const bool deformable,
                const bool modulated,
                at::Tensor &grad_mask) {
            if (input.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
                deform_conv2d_compute_grad_mask_cuda(columns,
                                                     input,
                                                     offset,
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
                                                     batch_sz,
                                                     offset_groups,
                                                     mask_groups,
                                                     deformable,
                                                     modulated,
                                                     grad_mask);
#else
                AT_ERROR("Not compiled with GPU support");
#endif
            } else {
                deform_conv2d_compute_grad_mask_cpu(columns,
                                                    input,
                                                    offset,
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
                                                    batch_sz,
                                                    offset_groups,
                                                    mask_groups,
                                                    deformable,
                                                    modulated,
                                                    grad_mask);
            }
        }
    }
}
