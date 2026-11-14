#include "ops/deform_conv1d.h"
#include "ops/deform_conv2d.h"
#include "ops/deform_conv3d.h"
#include <torch/csrc/stable/ops.h>
#include <iostream>
#include <vector>
#include "ops/deform_conv_transpose1d.h"
#include "ops/deform_conv_transpose2d.h"
#include "ops/deform_conv_transpose3d.h"

// Independent scatter reference exercises overlap, cropping, dilation and
// output_padding in every public transposed wrapper.
template <int Dim, typename Op>
bool check_transposed(Op op) {
    std::vector<int64_t> input_shape(Dim + 2, 3), weight_shape(Dim + 2, 2);
    input_shape[0] = input_shape[1] = weight_shape[0] = weight_shape[1] = 1;
    auto input = torch::stable::empty(input_shape, torch::headeronly::ScalarType::Double);
    auto weight = torch::stable::empty(weight_shape, torch::headeronly::ScalarType::Double);
    for (int i = 0; i < input.numel(); ++i) input.mutable_data_ptr<double>()[i] = i + 1;
    for (int k = 0; k < weight.numel(); ++k) weight.mutable_data_ptr<double>()[k] = k + 1;
    torch::stable::Tensor output = op(input, weight);
    if (output.dim() != Dim + 2 || output.size(0) != 1 || output.size(1) != 1) return false;
    int count = 1;
    for (int d = 0; d < Dim; ++d) {
        if (output.size(d + 2) != 6) return false;
        count *= 6;
    }
    std::vector<double> expected(count, 0);
    for (int i = 0; i < input.numel(); ++i) {
        for (int k = 0; k < weight.numel(); ++k) {
            int in_index = i, kernel_index = k, out_index = 0, factor = 1;
            bool inside = true;
            for (int d = Dim - 1; d >= 0; --d) {
                const int pos = (in_index % 3) * 2 - 1 + (kernel_index % 2) * 2;
                inside &= pos >= 0 && pos < 6;
                out_index += pos * factor;
                factor *= 6; in_index /= 3; kernel_index /= 2;
            }
            if (inside) expected[out_index] += (i + 1) * (k + 1);
        }
    }
    for (int i = 0; i < count; ++i)
        if (output.const_data_ptr<double>()[i] != expected[i]) return false;
    return true;
}

int main() {
    auto input = torch::stable::empty({1, 1, 3}, torch::headeronly::ScalarType::Double);
    auto weight = torch::stable::empty({1, 1, 2}, torch::headeronly::ScalarType::Double);
    for (int i = 0; i < 3; ++i) input.mutable_data_ptr<double>()[i] = i;
    for (int i = 0; i < 2; ++i) weight.mutable_data_ptr<double>()[i] = 1;
    auto output = tvdcn::ops::deform_conv1d(input, weight);
    const double expected1[] = {1, 3};
    if (output.dim() != 3 || output.size(2) != 2) return 1;
    for (int i = 0; i < 2; ++i)
        if (output.const_data_ptr<double>()[i] != expected1[i]) return 2;

    input = torch::stable::empty({1, 1, 3, 3}, torch::headeronly::ScalarType::Double);
    weight = torch::stable::empty({1, 1, 2, 2}, torch::headeronly::ScalarType::Double);
    for (int i = 0; i < 9; ++i) input.mutable_data_ptr<double>()[i] = i;
    for (int i = 0; i < 4; ++i) weight.mutable_data_ptr<double>()[i] = 1;
    output = tvdcn::ops::deform_conv2d(input, weight);
    const double expected2[] = {8, 12, 20, 24};
    if (output.dim() != 4 || output.size(2) != 2 || output.size(3) != 2) return 3;
    for (int i = 0; i < 4; ++i)
        if (output.const_data_ptr<double>()[i] != expected2[i]) return 4;

    input = torch::stable::empty({1, 1, 3, 3, 3}, torch::headeronly::ScalarType::Double);
    weight = torch::stable::empty({1, 1, 2, 2, 2}, torch::headeronly::ScalarType::Double);
    for (int i = 0; i < 27; ++i) input.mutable_data_ptr<double>()[i] = i;
    for (int i = 0; i < 8; ++i) weight.mutable_data_ptr<double>()[i] = 1;
    output = tvdcn::ops::deform_conv3d(input, weight);
    const double expected3[] = {52, 60, 76, 84, 124, 132, 148, 156};
    if (output.dim() != 5 || output.size(2) != 2 || output.size(3) != 2 || output.size(4) != 2) return 5;
    for (int i = 0; i < 8; ++i)
        if (output.const_data_ptr<double>()[i] != expected3[i]) return 6;
    if (!check_transposed<1>([](const auto& x, const auto& w) {
        return tvdcn::ops::deform_conv_transpose1d(x, w, {}, {}, {}, {2}, {1}, {1}, {2});
    })) return 7;
    if (!check_transposed<2>([](const auto& x, const auto& w) {
        return tvdcn::ops::deform_conv_transpose2d(x, w, {}, {}, {}, {2}, {1}, {1}, {2});
    })) return 8;
    if (!check_transposed<3>([](const auto& x, const auto& w) {
        return tvdcn::ops::deform_conv_transpose3d(x, w, {}, {}, {}, {2}, {1}, {1}, {2});
    })) return 9;
    std::cout << "C++ deform_conv/transpose1d/2d/3d CPU smoke passed\n";
}
