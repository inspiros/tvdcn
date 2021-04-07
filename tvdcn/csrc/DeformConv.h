#pragma once

#include "cpu/vision_cpu.h"

#if defined(WITH_CUDA)
#include "cuda/vision_cuda.h"
#endif

at::Tensor DeformConv1d_forward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	const int stride,
	const int padding,
	const int dilation,
	const int groups,
	const int offset_groups,
	const bool modulated) {
	if (input.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
		return DeformConv1d_forward_cuda(
			input.contiguous(),
			weight.contiguous(),
			offset.contiguous(),
			mask.contiguous(),
			bias.contiguous(),
			stride,
			padding,
			dilation,
			groups,
			offset_groups,
			modulated);
#else
		AT_ERROR("Not compiled with GPU support");
#endif
	}
	return DeformConv1d_forward_cpu(
		input.contiguous(),
		weight.contiguous(),
		offset.contiguous(),
		mask.contiguous(),
		bias.contiguous(),
		stride,
		padding,
		dilation,
		groups,
		offset_groups,
		modulated);
}

at::Tensor DeformConv2d_forward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	const std::pair<int, int>& stride,
	const std::pair<int, int>& padding,
	const std::pair<int, int>& dilation,
	const int groups,
	const int offset_groups,
	const bool modulated) {
	if (input.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
		return DeformConv2d_forward_cuda(
			input.contiguous(),
			weight.contiguous(),
			offset.contiguous(),
			mask.contiguous(),
			bias.contiguous(),
			stride,
			padding,
			dilation,
			groups,
			offset_groups,
			modulated);
#else
		AT_ERROR("Not compiled with GPU support");
#endif
	}
	return DeformConv2d_forward_cpu(
		input.contiguous(),
		weight.contiguous(),
		offset.contiguous(),
		mask.contiguous(),
		bias.contiguous(),
		stride,
		padding,
		dilation,
		groups,
		offset_groups,
		modulated);
}

at::Tensor DeformConv3d_forward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	const std::tuple<int, int, int>& stride,
	const std::tuple<int, int, int>& padding,
	const std::tuple<int, int, int>& dilation,
	const int groups,
	const int offset_groups,
	const bool modulated) {
	if (input.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
		return DeformConv3d_forward_cuda(
			input.contiguous(),
			weight.contiguous(),
			offset.contiguous(),
			mask.contiguous(),
			bias.contiguous(),
			stride,
			padding,
			dilation,
			groups,
			offset_groups,
			modulated);
#else
		AT_ERROR("Not compiled with GPU support");
#endif
	}
	return DeformConv3d_forward_cpu(
		input.contiguous(),
		weight.contiguous(),
		offset.contiguous(),
		mask.contiguous(),
		bias.contiguous(),
		stride,
		padding,
		dilation,
		groups,
		offset_groups,
		modulated);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> DeformConv1d_backward(
	const at::Tensor& grad,
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	const int stride,
	const int padding,
	const int dilation,
	const int groups,
	const int offset_groups,
	const bool modulated) {
	if (grad.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
		return DeformConv1d_backward_cuda(
			grad.contiguous(),
			input.contiguous(),
			weight.contiguous(),
			offset.contiguous(),
			mask.contiguous(),
			bias.contiguous(),
			stride,
			padding,
			dilation,
			groups,
			offset_groups,
			modulated);
#else
		AT_ERROR("Not compiled with GPU support");
#endif
	}
	return DeformConv1d_backward_cpu(
		grad.contiguous(),
		input.contiguous(),
		weight.contiguous(),
		offset.contiguous(),
		mask.contiguous(),
		bias.contiguous(),
		stride,
		padding,
		dilation,
		groups,
		offset_groups,
		modulated);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> DeformConv2d_backward(
	const at::Tensor& grad,
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	const std::pair<int, int>& stride,
	const std::pair<int, int>& padding,
	const std::pair<int, int>& dilation,
	const int groups,
	const int offset_groups,
	const bool modulated) {
	if (grad.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
		return DeformConv2d_backward_cuda(
			grad.contiguous(),
			input.contiguous(),
			weight.contiguous(),
			offset.contiguous(),
			mask.contiguous(),
			bias.contiguous(),
			stride,
			padding,
			dilation,
			groups,
			offset_groups,
			modulated);
#else
		AT_ERROR("Not compiled with GPU support");
#endif
	}
	return DeformConv2d_backward_cpu(
		grad.contiguous(),
		input.contiguous(),
		weight.contiguous(),
		offset.contiguous(),
		mask.contiguous(),
		bias.contiguous(),
		stride,
		padding,
		dilation,
		groups,
		offset_groups,
		modulated);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> DeformConv3d_backward(
	const at::Tensor& grad,
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	const std::tuple<int, int, int>& stride,
	const std::tuple<int, int, int>& padding,
	const std::tuple<int, int, int>& dilation,
	const int groups,
	const int offset_groups,
	const bool modulated) {
	if (grad.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
		return DeformConv3d_backward_cuda(
			grad.contiguous(),
			input.contiguous(),
			weight.contiguous(),
			offset.contiguous(),
			mask.contiguous(),
			bias.contiguous(),
			stride,
			padding,
			dilation,
			groups,
			offset_groups,
			modulated);
#else
		AT_ERROR("Not compiled with GPU support");
#endif
	}
	return DeformConv3d_backward_cpu(
		grad.contiguous(),
		input.contiguous(),
		weight.contiguous(),
		offset.contiguous(),
		mask.contiguous(),
		bias.contiguous(),
		stride,
		padding,
		dilation,
		groups,
		offset_groups,
		modulated);
}

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class DeformConv1dFunction
	: public torch::autograd::Function<DeformConv1dFunction> {
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
		int64_t dilation,
		int64_t groups,
		int64_t offset_groups,
		bool modulated) {
		auto output = DeformConv1d_forward(
			input,
			weight,
			offset,
			mask,
			bias,
			stride,
			pad,
			dilation,
			groups,
			offset_groups,
			modulated);

		ctx->save_for_backward({ input, weight, offset, mask, bias });
		ctx->saved_data["stride"] = stride;
		ctx->saved_data["pad"] = pad;
		ctx->saved_data["dilation"] = dilation;
		ctx->saved_data["groups"] = groups;
		ctx->saved_data["offset_groups"] = offset_groups;
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
		auto dilation = ctx->saved_data["dilation"].toInt();
		auto groups = ctx->saved_data["groups"].toInt();
		auto offset_groups = ctx->saved_data["offset_groups"].toInt();
		auto modulated = ctx->saved_data["modulated"].toBool();

		auto grads = DeformConv1d_backward(
			grad_output[0],
			input,
			weight,
			offset,
			mask,
			bias,
			stride,
			pad,
			dilation,
			groups,
			offset_groups,
			modulated);
		auto grad_input = std::get<0>(grads);
		auto grad_weight = std::get<1>(grads);
		auto grad_offset = std::get<2>(grads);
		auto grad_mask = modulated ? std::get<3>(grads) : Variable();
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
		};
	}
};

class DeformConv2dFunction
	: public torch::autograd::Function<DeformConv2dFunction> {
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
		int64_t dilation_h,
		int64_t dilation_w,
		int64_t groups,
		int64_t offset_groups,
		bool modulated) {
		auto output = DeformConv2d_forward(
			input,
			weight,
			offset,
			mask,
			bias,
			std::make_pair(stride_h, stride_w),
			std::make_pair(pad_h, pad_w),
			std::make_pair(dilation_h, dilation_w),
			groups,
			offset_groups,
			modulated);

		ctx->save_for_backward({ input, weight, offset, mask, bias });
		ctx->saved_data["stride_h"] = stride_h;
		ctx->saved_data["stride_w"] = stride_w;
		ctx->saved_data["pad_h"] = pad_h;
		ctx->saved_data["pad_w"] = pad_w;
		ctx->saved_data["dilation_h"] = dilation_h;
		ctx->saved_data["dilation_w"] = dilation_w;
		ctx->saved_data["groups"] = groups;
		ctx->saved_data["offset_groups"] = offset_groups;
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
		auto dilation_h = ctx->saved_data["dilation_h"].toInt();
		auto dilation_w = ctx->saved_data["dilation_w"].toInt();
		auto groups = ctx->saved_data["groups"].toInt();
		auto offset_groups = ctx->saved_data["offset_groups"].toInt();
		auto modulated = ctx->saved_data["modulated"].toBool();

		auto grads = DeformConv2d_backward(
			grad_output[0],
			input,
			weight,
			offset,
			mask,
			bias,
			std::make_pair(stride_h, stride_w),
			std::make_pair(pad_h, pad_w),
			std::make_pair(dilation_h, dilation_w),
			groups,
			offset_groups,
			modulated);
		auto grad_input = std::get<0>(grads);
		auto grad_weight = std::get<1>(grads);
		auto grad_offset = std::get<2>(grads);
		auto grad_mask = modulated ? std::get<3>(grads) : Variable();
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

class DeformConv3dFunction
	: public torch::autograd::Function<DeformConv3dFunction> {
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
		int64_t dilation_d,
		int64_t dilation_h,
		int64_t dilation_w,
		int64_t groups,
		int64_t offset_groups,
		bool modulated) {
		auto output = DeformConv3d_forward(
			input,
			weight,
			offset,
			mask,
			bias,
			std::make_tuple(stride_d, stride_h, stride_w),
			std::make_tuple(pad_d, pad_h, pad_w),
			std::make_tuple(dilation_d, dilation_h, dilation_w),
			groups,
			offset_groups,
			modulated);

		ctx->save_for_backward({ input, weight, offset, mask, bias });
		ctx->saved_data["stride_d"] = stride_d;
		ctx->saved_data["stride_h"] = stride_h;
		ctx->saved_data["stride_w"] = stride_w;
		ctx->saved_data["pad_d"] = pad_d;
		ctx->saved_data["pad_h"] = pad_h;
		ctx->saved_data["pad_w"] = pad_w;
		ctx->saved_data["dilation_d"] = dilation_d;
		ctx->saved_data["dilation_h"] = dilation_h;
		ctx->saved_data["dilation_w"] = dilation_w;
		ctx->saved_data["groups"] = groups;
		ctx->saved_data["offset_groups"] = offset_groups;
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
		auto dilation_d = ctx->saved_data["dilation_d"].toInt();
		auto dilation_h = ctx->saved_data["dilation_h"].toInt();
		auto dilation_w = ctx->saved_data["dilation_w"].toInt();
		auto groups = ctx->saved_data["groups"].toInt();
		auto offset_groups = ctx->saved_data["offset_groups"].toInt();
		auto modulated = ctx->saved_data["modulated"].toBool();

		auto grads = DeformConv3d_backward(
			grad_output[0],
			input,
			weight,
			offset,
			mask,
			bias,
			std::make_tuple(stride_d, stride_h, stride_w),
			std::make_tuple(pad_d, pad_h, pad_w),
			std::make_tuple(dilation_d, dilation_h, dilation_w),
			groups,
			offset_groups,
			modulated);
		auto grad_input = std::get<0>(grads);
		auto grad_weight = std::get<1>(grads);
		auto grad_offset = std::get<2>(grads);
		auto grad_mask = modulated ? std::get<3>(grads) : Variable();
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
		};
	}
};

at::Tensor deform_conv1d(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	int64_t stride,
	int64_t pad,
	int64_t dilation,
	int64_t groups,
	int64_t offset_groups,
	bool modulated) {
	auto result = DeformConv1dFunction::apply(
		input,
		weight,
		offset,
		mask,
		bias,
		stride,
		pad,
		dilation,
		groups,
		offset_groups,
        modulated);
	return result[0];
}

at::Tensor deform_conv2d(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset,
	const at::Tensor& mask,
	const at::Tensor& bias,
	int64_t stride_h,
	int64_t stride_w,
	int64_t pad_h,
	int64_t pad_w,
	int64_t dilation_h,
	int64_t dilation_w,
	int64_t groups,
	int64_t offset_groups,
	bool modulated) {
	auto result = DeformConv2dFunction::apply(
		input,
		weight,
		offset,
		mask,
		bias,
		stride_h,
		stride_w,
		pad_h,
		pad_w,
		dilation_h,
		dilation_w,
		groups,
		offset_groups,
		modulated);
	return result[0];
}

at::Tensor deform_conv3d(
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
	int64_t dilation_d,
	int64_t dilation_h,
	int64_t dilation_w,
	int64_t groups,
	int64_t offset_groups,
	bool modulated) {
	auto result = DeformConv3dFunction::apply(
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
		dilation_d,
		dilation_h,
		dilation_w,
		groups,
		offset_groups,
		modulated);
	return result[0];
}
