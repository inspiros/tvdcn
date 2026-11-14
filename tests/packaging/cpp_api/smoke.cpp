#include "ops/deform_conv1d.h"
#include "ops/deform_conv2d.h"
#include "ops/deform_conv3d.h"
#include <torch/csrc/stable/ops.h>
#include <iostream>

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
    std::cout << "C++ deform_conv1d/2d/3d CPU smoke passed\n";
}
