#pragma once

#include <ATen/TensorUtils.h>

#include "../../macros.h"

namespace at {
    TVDCN_API void checkSameDeviceType(CheckedFrom c, const TensorArg &t1, const TensorArg &t2);

    TVDCN_API void checkAllSameDeviceType(CheckedFrom c, ArrayRef<TensorArg> tensors);

    TVDCN_API void checkAllSameDeviceTypeExceptUndefined(CheckedFrom c, ArrayRef<TensorArg> tensors);

    TVDCN_API void checkSameDevice(CheckedFrom c, const TensorArg &t1, const TensorArg &t2);

    TVDCN_API void checkAllSameDevice(CheckedFrom c, ArrayRef<TensorArg> tensors);

    TVDCN_API void checkAllSameDeviceExceptUndefined(CheckedFrom c, ArrayRef<TensorArg> tensors);

    TVDCN_API void checkAllSameGPUExceptUndefined(CheckedFrom c, ArrayRef<TensorArg> tensors);

    TVDCN_API void checkAllSameTypeExceptUndefined(CheckedFrom c, ArrayRef<TensorArg> tensors);
}
