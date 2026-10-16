#pragma once
#include "types.hpp"
#include <cstdint>

namespace v12 {

struct BufferDesc {
    u32     binding;
    DType   dtype;
    u32     element_count;   // per-element
    u32     stride;          // bytes between elements
    bool    read_only;
    std::string name;
};

enum class Storage {
    GPU-Only,      // device local
    HostVisible,   // host visible + coherent
};

struct MemLayout {
    // All buffers use std430 layout in GLSL.
    // Vectors are 16-byte aligned. Scalars are 4-byte aligned.
    // Int8 values are packed 4-per-vec4 for dot-product lanes.
    static constexpr u32 VEC4_SIZE     = 16;
    static constexpr u32 F32_SIZE      = 4;
    static constexpr u32 I8_PER_VEC4   = 4;
    static constexpr u32 I8_PACK_SIZE  = 4;
    static constexpr u32 I32_SIZE      = 4;
};

} // namespace v12
