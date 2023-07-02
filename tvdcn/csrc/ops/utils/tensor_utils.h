#pragma once

#include <ATen/TensorUtils.h>

#include "../../macros.h"

namespace at {
    void checkSameDeviceType(CheckedFrom c, const TensorArg &t1, const TensorArg &t2);

    void checkAllSameDeviceType(CheckedFrom c, ArrayRef<TensorArg> tensors);

    void checkAllSameDeviceTypeExceptUndefined(CheckedFrom c, ArrayRef<TensorArg> tensors);

    void checkSameDevice(CheckedFrom c, const TensorArg &t1, const TensorArg &t2);

    void checkAllSameDevice(CheckedFrom c, ArrayRef<TensorArg> tensors);

    void checkAllSameDeviceExceptUndefined(CheckedFrom c, ArrayRef<TensorArg> tensors);

    void checkAllSameGPUExceptUndefined(CheckedFrom c, ArrayRef<TensorArg> tensors);

    void checkAllSameTypeExceptUndefined(CheckedFrom c, ArrayRef<TensorArg> tensors);
}
