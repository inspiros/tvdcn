#pragma once

#include "ops/ops.h"
#include "macros.h"

namespace tvdcn {
    TVDCN_API int64_t cuda_version();
    TVDCN_API std::string cuda_arch_flags();
}
