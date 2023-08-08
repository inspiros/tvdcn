#include <ATen/TensorUtils.h>

namespace at {
    static inline std::vector<int64_t> definedPositions(ArrayRef<TensorArg> tensors) {
        std::vector<int64_t> res;
        res.reserve(tensors.size());
        for (const auto i: c10::irange(tensors.size())) {
            if (tensors[i]->defined())
                res.emplace_back(i);
        }
        return res;
    }

    static void checkDeviceType(CheckedFrom c,
                                const TensorArg &t,
                                DeviceType device_type) {
        TORCH_CHECK(
                !t->defined() || t->device().type() == device_type,
                "Expected tensor for argument #",
                t.pos,
                " '",
                t.name,
                "' to have ",
                device_type,
                " DeviceType, but got tensor with ",
                t->device().type(),
                " DeviceType (while checking arguments for ", c, ")")
    }

    void checkDeviceType(
            CheckedFrom c,
            ArrayRef<TensorArg> tensors,
            DeviceType device_type) {
        for (const auto i: c10::irange(tensors.size())) {
            checkDeviceType(c, tensors[i], device_type);
        }
    }

    void checkDeviceTypeExceptUndefined(
            CheckedFrom c,
            ArrayRef<TensorArg> tensors,
            DeviceType device_type) {
        auto defined_pos = definedPositions(tensors);
        for (const auto i: defined_pos) {
            checkDeviceType(c, tensors[i], device_type);
        }
    }

    void checkSameDeviceType(
            CheckedFrom c,
            const TensorArg &t1,
            const TensorArg &t2) {
        TORCH_CHECK(t1->device().type() == t2->device().type(),
                    "Expected tensor for argument #",
                    t1.pos,
                    " '",
                    t1.name,
                    "' to have the same device type as tensor for argument #",
                    t2.pos,
                    " '",
                    t2.name,
                    "'; but device type ",
                    t1->device().type(),
                    " does not equal ",
                    t2->device().type(),
                    " (while checking arguments for ",
                    c,
                    ")")
    }

    void checkAllSameDeviceType(
            CheckedFrom c,
            ArrayRef<TensorArg> tensors) {
        if (tensors.size() < 2)
            return;
        for (const auto i: c10::irange(1, tensors.size())) {
            checkSameDeviceType(c, tensors[0], tensors[i]);
        }
    }

    void checkAllSameDeviceTypeExceptUndefined(
            CheckedFrom c,
            ArrayRef<TensorArg> tensors) {
        auto defined_pos = definedPositions(tensors);
        if (defined_pos.size() < 2)
            return;
        auto t0 = tensors[defined_pos[0]];
        for (auto iter = std::next(defined_pos.begin()); iter != defined_pos.end(); iter++) {
            checkSameDeviceType(c, t0, tensors[*iter]);
        }
    }

    void checkSameDevice(
            CheckedFrom c,
            const TensorArg &t1,
            const TensorArg &t2) {
        TORCH_CHECK(t1->device() == t2->device(),
                    "Expected tensor for argument #",
                    t1.pos,
                    " '",
                    t1.name,
                    "' to have the same device as tensor for argument #",
                    t2.pos,
                    " '",
                    t2.name,
                    "'; but device ",
                    t1->device(),
                    " does not equal ",
                    t2->device(),
                    " (while checking arguments for ",
                    c,
                    ")")
    }

    void checkAllSameDevice(
            CheckedFrom c,
            ArrayRef<TensorArg> tensors) {
        if (tensors.size() < 2)
            return;
        for (const auto i: c10::irange(1, tensors.size())) {
            checkSameDevice(c, tensors[0], tensors[i]);
        }
    }

    void checkAllSameDeviceExceptUndefined(
            CheckedFrom c,
            ArrayRef<TensorArg> tensors) {
        auto defined_pos = definedPositions(tensors);
        if (defined_pos.size() < 2)
            return;
        auto t0 = tensors[defined_pos[0]];
        for (auto iter = std::next(defined_pos.begin()); iter != defined_pos.end(); iter++) {
            checkSameDevice(c, t0, tensors[*iter]);
        }
    }

    void checkAllSameGPUExceptUndefined(
            CheckedFrom c,
            ArrayRef<TensorArg> tensors) {
        auto defined_pos = definedPositions(tensors);
        if (defined_pos.size() < 2)
            return;
        auto t0 = tensors[defined_pos[0]];
        for (auto iter = std::next(defined_pos.begin()); iter != defined_pos.end(); iter++) {
            checkSameGPU(c, t0, tensors[*iter]);
        }
    }

    void checkAllSameTypeExceptUndefined(CheckedFrom c, ArrayRef<TensorArg> tensors) {
        auto defined_pos = definedPositions(tensors);
        if (defined_pos.size() < 2)
            return;
        auto t0 = tensors[defined_pos[0]];
        for (auto iter = std::next(defined_pos.begin()); iter != defined_pos.end(); iter++) {
            checkSameType(c, t0, tensors[*iter]);
        }
    }
}
