// Common push constant — 128 bytes, explicit offsets for C++ host packing.
// Included via: #include "pc.h"  (compiled with glslc -I shaders/)
// All shaders (fused, unfused, tiled) share this layout.
layout(push_constant) uniform Params {
    layout(offset =   0) uint  count;
    layout(offset =   4) uint  width;
    layout(offset =   8) uint  in_w;
    layout(offset =  12) uint  in_h;
    layout(offset =  16) vec4  row0;
    layout(offset =  32) vec4  row1;
    layout(offset =  48) vec4  row2;
    layout(offset =  64) vec4  row3;
    layout(offset =  80) vec4  bias;
    layout(offset =  96) vec4  scale;
    layout(offset = 112) vec4  sbbias;
} pc;
