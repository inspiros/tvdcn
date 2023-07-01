#include <ATen/TensorUtils.h>

namespace at {
    namespace {
        inline std::vector<int> definedPositions(ArrayRef<TensorArg> tensors) {
            std::vector<int> res = {};
            for (int i = 0; i < tensors.size(); i++) {
                if (tensors[i]->defined())
                    res.push_back(i);
            }
            return res;
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
        for (int i = 1; i < tensors.size(); i++) {
            checkSameDeviceType(c, tensors[0], tensors[i]);
        }
    }

    void checkAllSameDeviceTypeExceptUndefined(
            CheckedFrom c,
            ArrayRef<TensorArg> tensors) {
        auto defined_pos = definedPositions(tensors);
        if (defined_pos.size() < 2)
            return;
        for (int i = 1; i < defined_pos.size(); i++) {
            checkSameDeviceType(c, tensors[defined_pos[0]], tensors[defined_pos[i]]);
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
        for (int i = 1; i < tensors.size(); i++) {
            checkSameDevice(c, tensors[0], tensors[i]);
        }
    }

    void checkAllSameDeviceExceptUndefined(
            CheckedFrom c,
            ArrayRef<TensorArg> tensors) {
        auto defined_pos = definedPositions(tensors);
        if (defined_pos.size() < 2)
            return;
        for (int i = 1; i < defined_pos.size(); i++) {
            checkSameDevice(c, tensors[defined_pos[0]], tensors[defined_pos[i]]);
        }
    }

    void checkAllSameGPUExceptUndefined(
            CheckedFrom c,
            ArrayRef<TensorArg> tensors) {
        auto defined_pos = definedPositions(tensors);
        if (defined_pos.size() < 2)
            return;
        for (int i = 1; i < defined_pos.size(); i++) {
            checkSameGPU(c, tensors[defined_pos[0]], tensors[defined_pos[i]]);
        }
    }

    void checkAllSameTypeExceptUndefined(CheckedFrom c, ArrayRef<TensorArg> tensors) {
        auto defined_pos = definedPositions(tensors);
        if (defined_pos.size() < 2)
            return;
        for (int i = 1; i < defined_pos.size(); i++) {
            checkSameType(c, tensors[defined_pos[0]], tensors[defined_pos[i]]);
        }
    }
}
