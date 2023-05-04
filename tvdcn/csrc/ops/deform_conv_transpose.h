#pragma once

#include <ATen/Tensor.h>

at::Tensor deform_conv_transpose1d_forward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	int stride,
	int padding,
	int output_padding,
	int dilation,
	int groups,
	int offset_groups,
    int mask_groups,
	bool deformable,
	bool modulated);

at::Tensor deform_conv_transpose2d_forward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	const std::pair<int, int>& stride,
	const std::pair<int, int>& padding,
	const std::pair<int, int>& output_padding,
	const std::pair<int, int>& dilation,
	int groups,
	int offset_groups,
    int mask_groups,
    bool deformable,
	bool modulated);

at::Tensor deform_conv_transpose3d_forward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	const std::tuple<int, int, int>& stride,
	const std::tuple<int, int, int>& padding,
	const std::tuple<int, int, int>& output_padding,
	const std::tuple<int, int, int>& dilation,
	int groups,
	int offset_groups,
	int mask_groups,
    bool deformable,
	bool modulated);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv_transpose1d_backward(
	const at::Tensor& grad,
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	int stride,
	int padding,
	int output_padding,
	int dilation,
	int groups,
	int offset_groups,
	int mask_groups,
    bool deformable,
	bool modulated);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv_transpose2d_backward(
	const at::Tensor& grad,
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	const std::pair<int, int>& stride,
	const std::pair<int, int>& padding,
	const std::pair<int, int>& output_padding,
	const std::pair<int, int>& dilation,
	int groups,
	int offset_groups,
	int mask_groups,
    bool deformable,
	bool modulated);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv_transpose3d_backward(
	const at::Tensor& grad,
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	const std::tuple<int, int, int>& stride,
	const std::tuple<int, int, int>& padding,
	const std::tuple<int, int, int>& output_padding,
	const std::tuple<int, int, int>& dilation,
	int groups,
	int offset_groups,
	int mask_groups,
    bool deformable,
	bool modulated);

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class DeformConvTranspose1dFunction
	: public torch::autograd::Function<DeformConvTranspose1dFunction> {
public:
	static variable_list forward(
		AutogradContext* ctx,
		Variable input,
		Variable weight,
		Variable offset,
		Variable mask,
		Variable bias,
		int64_t stride,
		int64_t pad,
		int64_t out_pad,
		int64_t dilation,
		int64_t groups,
		int64_t offset_groups,
		int64_t mask_groups,
		bool deformable,
		bool modulated) {
		auto output = deform_conv_transpose1d_forward(
                input,
                weight,
                offset,
                mask,
                bias,
                stride,
                pad,
                out_pad,
                dilation,
                groups,
                offset_groups,
                mask_groups,
                deformable,
                modulated);

		ctx->save_for_backward({ input, weight, offset, mask, bias });
		ctx->saved_data["stride"] = stride;
		ctx->saved_data["pad"] = pad;
		ctx->saved_data["out_pad"] = out_pad;
		ctx->saved_data["dilation"] = dilation;
		ctx->saved_data["groups"] = groups;
		ctx->saved_data["offset_groups"] = offset_groups;
		ctx->saved_data["mask_groups"] = mask_groups;
		ctx->saved_data["deformable"] = deformable;
		ctx->saved_data["modulated"] = modulated;

		return {
			output,
		};
	}

	static variable_list backward(
		AutogradContext* ctx,
		variable_list grad_output) {
		auto saved = ctx->get_saved_variables();
		auto input = saved[0];
		auto weight = saved[1];
		auto offset = saved[2];
		auto mask = saved[3];
		auto bias = saved[4];

		auto stride = ctx->saved_data["stride"].toInt();
		auto pad = ctx->saved_data["pad"].toInt();
		auto out_pad = ctx->saved_data["out_pad"].toInt();
		auto dilation = ctx->saved_data["dilation"].toInt();
		auto groups = ctx->saved_data["groups"].toInt();
		auto offset_groups = ctx->saved_data["offset_groups"].toInt();
		auto mask_groups = ctx->saved_data["mask_groups"].toInt();
		auto deformable = ctx->saved_data["deformable"].toBool();
		auto modulated = ctx->saved_data["modulated"].toBool();

		auto grads = deform_conv_transpose1d_backward(
                grad_output[0],
                input,
                weight,
                offset,
                mask,
                bias,
                stride,
                pad,
                out_pad,
                dilation,
                groups,
                offset_groups,
                mask_groups,
                deformable,
                modulated);
		auto grad_input = std::get<0>(grads);
		auto grad_weight = std::get<1>(grads);
		auto grad_offset = std::get<2>(grads);
		auto grad_mask = std::get<3>(grads);
		auto grad_bias = std::get<4>(grads);

		return {
			grad_input,
			grad_weight,
			grad_offset,
			grad_mask,
			grad_bias,
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
		};
	}
};

class DeformConvTranspose2dFunction
	: public torch::autograd::Function<DeformConvTranspose2dFunction> {
public:
	static variable_list forward(
		AutogradContext* ctx,
		Variable input,
		Variable weight,
		Variable offset,
		Variable mask,
		Variable bias,
		int64_t stride_h,
		int64_t stride_w,
		int64_t pad_h,
		int64_t pad_w,
		int64_t out_pad_h,
		int64_t out_pad_w,
		int64_t dilation_h,
		int64_t dilation_w,
		int64_t groups,
		int64_t offset_groups,
		int64_t mask_groups,
		bool deformable,
		bool modulated) {
		auto output = deform_conv_transpose2d_forward(
                input,
                weight,
                offset,
                mask,
                bias,
                std::make_pair(stride_h, stride_w),
                std::make_pair(pad_h, pad_w),
                std::make_pair(out_pad_h, out_pad_w),
                std::make_pair(dilation_h, dilation_w),
                groups,
                offset_groups,
                mask_groups,
                deformable,
                modulated);

		ctx->save_for_backward({ input, weight, offset, mask, bias });
		ctx->saved_data["stride_h"] = stride_h;
		ctx->saved_data["stride_w"] = stride_w;
		ctx->saved_data["pad_h"] = pad_h;
		ctx->saved_data["pad_w"] = pad_w;
		ctx->saved_data["out_pad_h"] = out_pad_h;
		ctx->saved_data["out_pad_w"] = out_pad_w;
		ctx->saved_data["dilation_h"] = dilation_h;
		ctx->saved_data["dilation_w"] = dilation_w;
		ctx->saved_data["groups"] = groups;
		ctx->saved_data["offset_groups"] = offset_groups;
		ctx->saved_data["mask_groups"] = mask_groups;
		ctx->saved_data["deformable"] = deformable;
		ctx->saved_data["modulated"] = modulated;

		return {
			output,
		};
	}

	static variable_list backward(
		AutogradContext* ctx,
		variable_list grad_output) {
		auto saved = ctx->get_saved_variables();
		auto input = saved[0];
		auto weight = saved[1];
		auto offset = saved[2];
		auto mask = saved[3];
		auto bias = saved[4];

		auto stride_h = ctx->saved_data["stride_h"].toInt();
		auto stride_w = ctx->saved_data["stride_w"].toInt();
		auto pad_h = ctx->saved_data["pad_h"].toInt();
		auto pad_w = ctx->saved_data["pad_w"].toInt();
		auto out_pad_h = ctx->saved_data["out_pad_h"].toInt();
		auto out_pad_w = ctx->saved_data["out_pad_w"].toInt();
		auto dilation_h = ctx->saved_data["dilation_h"].toInt();
		auto dilation_w = ctx->saved_data["dilation_w"].toInt();
		auto groups = ctx->saved_data["groups"].toInt();
		auto offset_groups = ctx->saved_data["offset_groups"].toInt();
		auto mask_groups = ctx->saved_data["mask_groups"].toInt();
		auto deformable = ctx->saved_data["deformable"].toBool();
		auto modulated = ctx->saved_data["modulated"].toBool();

		auto grads = deform_conv_transpose2d_backward(
                grad_output[0],
                input,
                weight,
                offset,
                mask,
                bias,
                std::make_pair(stride_h, stride_w),
                std::make_pair(pad_h, pad_w),
                std::make_pair(out_pad_h, out_pad_w),
                std::make_pair(dilation_h, dilation_w),
                groups,
                offset_groups,
                mask_groups,
                deformable,
                modulated);
		auto grad_input = std::get<0>(grads);
		auto grad_weight = std::get<1>(grads);
		auto grad_offset = std::get<2>(grads);
		auto grad_mask = std::get<3>(grads);
		auto grad_bias = std::get<4>(grads);

		return {
			grad_input,
			grad_weight,
			grad_offset,
			grad_mask,
			grad_bias,
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
		};
	}
};

class DeformConvTranspose3dFunction
	: public torch::autograd::Function<DeformConvTranspose3dFunction> {
public:
	static variable_list forward(
		AutogradContext* ctx,
		Variable input,
		Variable weight,
		Variable offset,
		Variable mask,
		Variable bias,
		int64_t stride_d,
		int64_t stride_h,
		int64_t stride_w,
		int64_t pad_d,
		int64_t pad_h,
		int64_t pad_w,
		int64_t out_pad_d,
		int64_t out_pad_h,
		int64_t out_pad_w,
		int64_t dilation_d,
		int64_t dilation_h,
		int64_t dilation_w,
		int64_t groups,
		int64_t offset_groups,
		int64_t mask_groups,
		bool deformable,
		bool modulated) {
		auto output = deform_conv_transpose3d_forward(
                input,
                weight,
                offset,
                mask,
                bias,
                std::make_tuple(stride_d, stride_h, stride_w),
                std::make_tuple(pad_d, pad_h, pad_w),
                std::make_tuple(out_pad_d, out_pad_h, out_pad_w),
                std::make_tuple(dilation_d, dilation_h, dilation_w),
                groups,
                offset_groups,
                mask_groups,
                deformable,
                modulated);

		ctx->save_for_backward({ input, weight, offset, mask, bias });
		ctx->saved_data["stride_d"] = stride_d;
		ctx->saved_data["stride_h"] = stride_h;
		ctx->saved_data["stride_w"] = stride_w;
		ctx->saved_data["pad_d"] = pad_d;
		ctx->saved_data["pad_h"] = pad_h;
		ctx->saved_data["pad_w"] = pad_w;
		ctx->saved_data["out_pad_d"] = out_pad_d;
		ctx->saved_data["out_pad_h"] = out_pad_h;
		ctx->saved_data["out_pad_w"] = out_pad_w;
		ctx->saved_data["dilation_d"] = dilation_d;
		ctx->saved_data["dilation_h"] = dilation_h;
		ctx->saved_data["dilation_w"] = dilation_w;
		ctx->saved_data["groups"] = groups;
		ctx->saved_data["offset_groups"] = offset_groups;
		ctx->saved_data["mask_groups"] = mask_groups;
		ctx->saved_data["deformable"] = deformable;
		ctx->saved_data["modulated"] = modulated;

		return {
			output,
		};
	}

	static variable_list backward(
		AutogradContext* ctx,
		variable_list grad_output) {
		auto saved = ctx->get_saved_variables();
		auto input = saved[0];
		auto weight = saved[1];
		auto offset = saved[2];
		auto mask = saved[3];
		auto bias = saved[4];

		auto stride_d = ctx->saved_data["stride_d"].toInt();
		auto stride_h = ctx->saved_data["stride_h"].toInt();
		auto stride_w = ctx->saved_data["stride_w"].toInt();
		auto pad_d = ctx->saved_data["pad_d"].toInt();
		auto pad_h = ctx->saved_data["pad_h"].toInt();
		auto pad_w = ctx->saved_data["pad_w"].toInt();
		auto out_pad_d = ctx->saved_data["out_pad_d"].toInt();
		auto out_pad_h = ctx->saved_data["out_pad_h"].toInt();
		auto out_pad_w = ctx->saved_data["out_pad_w"].toInt();
		auto dilation_d = ctx->saved_data["dilation_d"].toInt();
		auto dilation_h = ctx->saved_data["dilation_h"].toInt();
		auto dilation_w = ctx->saved_data["dilation_w"].toInt();
		auto groups = ctx->saved_data["groups"].toInt();
		auto offset_groups = ctx->saved_data["offset_groups"].toInt();
		auto mask_groups = ctx->saved_data["mask_groups"].toInt();
		auto deformable = ctx->saved_data["deformable"].toBool();
		auto modulated = ctx->saved_data["modulated"].toBool();

		auto grads = deform_conv_transpose3d_backward(
                grad_output[0],
                input,
                weight,
                offset,
                mask,
                bias,
                std::make_tuple(stride_d, stride_h, stride_w),
                std::make_tuple(pad_d, pad_h, pad_w),
                std::make_tuple(out_pad_d, out_pad_h, out_pad_w),
                std::make_tuple(dilation_d, dilation_h, dilation_w),
                groups,
                offset_groups,
                mask_groups,
                deformable,
                modulated);
		auto grad_input = std::get<0>(grads);
		auto grad_weight = std::get<1>(grads);
		auto grad_offset = std::get<2>(grads);
		auto grad_mask = std::get<3>(grads);
		auto grad_bias = std::get<4>(grads);

		return {
			grad_input,
			grad_weight,
			grad_offset,
			grad_mask,
			grad_bias,
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
			Variable(),
		};
	}
};

at::Tensor deform_conv_transpose1d(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	int64_t stride,
	int64_t pad,
	int64_t out_pad,
	int64_t dilation,
	int64_t groups,
	int64_t offset_groups,
	int64_t mask_groups,
	bool deformable,
	bool modulated) {
	auto result = DeformConvTranspose1dFunction::apply(
		input,
		weight,
		offset,
		mask,
		bias,
		stride,
		pad,
		out_pad,
		dilation,
		groups,
		offset_groups,
        mask_groups,
        deformable,
        modulated);
	return result[0];
}

at::Tensor deform_conv_transpose2d(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	int64_t stride_h,
	int64_t stride_w,
	int64_t pad_h,
	int64_t pad_w,
	int64_t out_pad_h,
	int64_t out_pad_w,
	int64_t dilation_h,
	int64_t dilation_w,
	int64_t groups,
	int64_t offset_groups,
	int64_t mask_groups,
	bool deformable,
	bool modulated) {
	auto result = DeformConvTranspose2dFunction::apply(
		input,
		weight,
		offset,
		mask,
		bias,
		stride_h,
		stride_w,
		pad_h,
		pad_w,
		out_pad_h,
        out_pad_w,
		dilation_h,
		dilation_w,
		groups,
		offset_groups,
		mask_groups,
        deformable,
		modulated);
	return result[0];
}

at::Tensor deform_conv_transpose3d(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	int64_t stride_d,
	int64_t stride_h,
	int64_t stride_w,
	int64_t pad_d,
	int64_t pad_h,
	int64_t pad_w,
	int64_t out_pad_d,
	int64_t out_pad_h,
	int64_t out_pad_w,
	int64_t dilation_d,
	int64_t dilation_h,
	int64_t dilation_w,
	int64_t groups,
	int64_t offset_groups,
	int64_t mask_groups,
	bool deformable,
	bool modulated) {
	auto result = DeformConvTranspose3dFunction::apply(
		input,
		weight,
		offset,
		mask,
		bias,
		stride_d,
		stride_h,
		stride_w,
		pad_d,
		pad_h,
		pad_w,
		out_pad_d,
        out_pad_h,
        out_pad_w,
		dilation_d,
		dilation_h,
		dilation_w,
		groups,
		offset_groups,
		mask_groups,
        deformable,
		modulated);
	return result[0];
}
