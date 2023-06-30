#include "../cpu/deform_conv1d_kernels_cpu.h"

#if defined(WITH_CUDA)

#include "../cuda/deform_conv1d_kernels_cuda.h"

#endif

namespace tvdcn {
    namespace ops {
        void arr2col(
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
            if (input.device().is_cuda()) {
#if defined(WITH_CUDA)
                arr2col_cuda(input,
                             offset,
                             mask,
                             in_channels,
                             width,
                             weight_w,
                             stride_w,
                             pad_w,
                             dilation_w,
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
                arr2col_cpu(input,
                            offset,
                            mask,
                            in_channels,
                            width,
                            weight_w,
                            stride_w,
                            pad_w,
                            dilation_w,
                            out_w,
                            batch_sz,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            columns);
            }
        }

        void col2arr(
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
            if (grad_input.device().is_cuda()) {
#if defined(WITH_CUDA)
                col2arr_cuda(columns,
                             offset,
                             mask,
                             in_channels,
                             width,
                             weight_w,
                             stride_w,
                             pad_w,
                             dilation_w,
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
                col2arr_cpu(columns,
                            offset,
                            mask,
                            in_channels,
                            width,
                            weight_w,
                            stride_w,
                            pad_w,
                            dilation_w,
                            out_w,
                            batch_sz,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            grad_input);
            }
        }

        void deform_conv1d_compute_grad_offset(
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
            if (input.device().is_cuda()) {
#if defined(WITH_CUDA)
                deform_conv1d_compute_grad_offset_cuda(columns,
                                                       input,
                                                       offset,
                                                       mask,
                                                       in_channels,
                                                       width,
                                                       weight_w,
                                                       stride_w,
                                                       pad_w,
                                                       dilation_w,
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
                deform_conv1d_compute_grad_offset_cpu(columns,
                                                      input,
                                                      offset,
                                                      mask,
                                                      in_channels,
                                                      width,
                                                      weight_w,
                                                      stride_w,
                                                      pad_w,
                                                      dilation_w,
                                                      out_w,
                                                      batch_sz,
                                                      offset_groups,
                                                      mask_groups,
                                                      deformable,
                                                      modulated,
                                                      grad_offset);
            }
        }

        void deform_conv1d_compute_grad_mask(
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
            if (input.device().is_cuda()) {
#if defined(WITH_CUDA)
                deform_conv1d_compute_grad_mask_cuda(columns,
                                                     input,
                                                     offset,
                                                     in_channels,
                                                     width,
                                                     weight_w,
                                                     stride_w,
                                                     pad_w,
                                                     dilation_w,
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
                deform_conv1d_compute_grad_mask_cpu(columns,
                                                    input,
                                                    offset,
                                                    in_channels,
                                                    width,
                                                    weight_w,
                                                    stride_w,
                                                    pad_w,
                                                    dilation_w,
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
