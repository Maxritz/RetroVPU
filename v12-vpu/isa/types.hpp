#pragma once
#include <cstdint>
#include <cstddef>
#include <array>

namespace v12 {

using f32 = float;
using f64 = double;
using i32 = int32_t;
using i64 = int64_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8  = int8_t;

struct Vec4 {
    f32 x, y, z, w;
    f32& operator[](size_t i) { return (&x)[i]; }
    const f32& operator[](size_t i) const { return (&x)[i]; }
};

static_assert(sizeof(Vec4) == 16, "Vec4 must be 16 bytes");

enum class DType : u32 {
    Float32 = 0,
    Int8    = 1,
    Int32   = 2,
    UInt32  = 3,
    Vec4F32 = 4,
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Int8:    return 1;
        case DType::Int32:   return 4;
        case DType::UInt32:  return 4;
        case DType::Vec4F32: return 16;
    }
    return 0;
}

} // namespace v12
