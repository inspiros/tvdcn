#include <ATen/autocast_mode.h>
#include <torch/types.h>

#include "../deform_conv_transpose.h"

namespace tvdcn {
    namespace ops {
        namespace {
            at::Tensor deform_conv_transpose3d_autocast(
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    at::IntArrayRef output_padding,
                    at::IntArrayRef dilation,
                    int64_t groups) {
                c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
                return deform_conv_transpose3d(
                        at::autocast::cached_cast(at::kFloat, input),
                        at::autocast::cached_cast(at::kFloat, weight),
                        at::autocast::cached_cast(at::kFloat, offset),
                        at::autocast::cached_cast(at::kFloat, mask),
                        at::autocast::cached_cast(at::kFloat, bias),
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        groups)
                        .to(input.scalar_type());
            }
        }

        TORCH_LIBRARY_IMPL(tvdcn, Autocast, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("tvdcn::deform_conv_transpose3d"),
                    TORCH_FN(deform_conv_transpose3d_autocast));
        }
    }
}
