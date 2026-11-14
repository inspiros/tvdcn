#include "ops/deform_conv2d.h"
#include <torch/csrc/stable/ops.h>
#include <iostream>

int main() {
    auto input = torch::stable::empty({1, 1, 3, 3}, torch::headeronly::ScalarType::Double);
    auto weight = torch::stable::empty({1, 1, 2, 2}, torch::headeronly::ScalarType::Double);
    for (int i = 0; i < 9; ++i) input.mutable_data_ptr<double>()[i] = i;
    for (int i = 0; i < 4; ++i) weight.mutable_data_ptr<double>()[i] = 1;
    auto output = tvdcn::ops::deform_conv2d(input, weight);
    const double expected[] = {8, 12, 20, 24};
    if (output.dim() != 4 || output.size(2) != 2 || output.size(3) != 2) return 1;
    for (int i = 0; i < 4; ++i)
        if (output.const_data_ptr<double>()[i] != expected[i]) return 2;
    std::cout << "C++ deform_conv2d CPU smoke passed\n";
}
