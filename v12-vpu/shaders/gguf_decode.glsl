// Clean-room GGUF quant decode for GPU (no dequant): each element's integer
// weight code and its sub-block (scale,min) are produced from the raw GGUF
// block bytes. Provenance (AGENTS.md): byte layouts are the public GGUF/ggml
// format spec we must match to read real files; all code here is original.
//
// Select type at compile time: glslc -DGGUF_TYPE=<id>  (2,6,8,10,11,12,13,14,41)
// Buffers: 0=weights(uint raw bytes) 1=actcodes(uint,int8 packed) 2=actscale(float)
//          3=out(float)
#include "pc.h"

layout(set=0, binding=0, std430) readonly  buffer W { uint wq[]; };
layout(set=0, binding=1, std430) readonly  buffer A { uint aq[]; };
layout(set=0, binding=2, std430) readonly  buffer S { float ad[]; };
layout(set=0, binding=3, std430) writeonly buffer O { float outv[]; };

#ifndef GGUF_TYPE
#define GGUF_TYPE 12
#endif

#if   GGUF_TYPE == 2
  #define BLK 32
  #define BPB 18
#elif GGUF_TYPE == 6
  #define BLK 32
  #define BPB 22
#elif GGUF_TYPE == 8
  #define BLK 32
  #define BPB 34
#elif GGUF_TYPE == 41
  #define BLK 128
  #define BPB 18
#elif GGUF_TYPE == 12
  #define BLK 256
  #define BPB 144
#elif GGUF_TYPE == 13
  #define BLK 256
  #define BPB 176
#elif GGUF_TYPE == 14
  #define BLK 256
  #define BPB 210
#elif GGUF_TYPE == 10
  #define BLK 256
  #define BPB 84
#elif GGUF_TYPE == 11
  #define BLK 256
  #define BPB 110
#else
  #error unsupported GGUF_TYPE
#endif

#define GROUP 16u

uint rd8(uint off) {
    return (wq[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}
uint rd32(uint off) {
    return rd8(off) | (rd8(off + 1u) << 8u) | (rd8(off + 2u) << 16u) | (rd8(off + 3u) << 24u);
}
float rdf16(uint off) {
    return unpackHalf2x16(rd8(off) | (rd8(off + 1u) << 8u)).x;
}
float scf8(uint off) {                 // signed int8 scale/quant byte
    int b = int(rd8(off));
    return float(b < 128 ? b : b - 256);
}
int a8(uint e) {                       // activation int8 code
    uint v = (aq[e >> 2u] >> ((e & 3u) * 8u)) & 0xFFu;
    return int(v < 128u ? v : v - 256u);
}

void k4sm(uint j, uint sb, out float sc, out float mn) {
    if (j < 4u) {
        sc = float(rd8(sb + j) & 0x3Fu);
        mn = float(rd8(sb + j + 4u) & 0x3Fu);
    } else {
        uint b = rd8(sb + j + 4u);
        uint c = rd8(sb + j - 4u);
        uint d2 = rd8(sb + j);
        sc = float((b & 0xFu) | ((c >> 6u) << 4u));
        mn = float((b >> 4u) | ((d2 >> 6u) << 4u));
    }
}
int q3s(uint j, uint sb) {             // Q3_K recovered signed sub-block scale
    uint u0 = rd32(sb), u1 = rd32(sb + 4u), t = rd32(sb + 8u);
    uint km1 = 0x03030303u, km2 = 0x0F0F0F0Fu;
    uint a0 = (u0 & km2)          | ((((t >> 0u) & km1)) << 4u);
    uint a1 = (u1 & km2)          | ((((t >> 2u) & km1)) << 4u);
    uint a2 = ((u0 >> 4u) & km2)  | ((((t >> 4u) & km1)) << 4u);
    uint a3 = ((u1 >> 4u) & km2)  | ((((t >> 6u) & km1)) << 4u);
    uint w = (j >> 2u) == 0u ? a0 : (j >> 2u) == 1u ? a1 : (j >> 2u) == 2u ? a2 : a3;
    uint by = (w >> ((j & 3u) * 8u)) & 0xFFu;
    return int(by < 128u ? by : by - 256u);
}

// Decode integer weight code + (scale,min) for element e of a row at byte rb.
void wdec(uint e, uint rb, out int code, out float wsc, out float wmn) {
    wsc = 0.0; wmn = 0.0;
#if GGUF_TYPE == 8                                  // Q8_0
    uint i = e >> 5u;
    uint base = rb + i * 34u;
    float d = rdf16(base);
    int b = int(rd8(base + 2u + (e & 31u)));
    code = b < 128 ? b : b - 256;
    wsc = d;
#elif GGUF_TYPE == 2                                // Q4_0
    uint base = rb + (e >> 5u) * 18u;
    float d = rdf16(base);
    uint lp = e & 31u;
    uint by = rd8(base + 2u + (lp & 15u));
    uint nib = lp < 16u ? (by & 0xFu) : (by >> 4u);
    code = int(nib);
    wsc = d; wmn = 8.0 * d;
#elif GGUF_TYPE == 6                                // Q5_0
    uint base = rb + (e >> 5u) * 22u;
    float d = rdf16(base);
    uint qh = rd32(base + 2u);
    uint lp = e & 31u;
    uint j = lp & 15u;
    uint by = rd8(base + 6u + j);
    uint hb = lp < 16u ? ((qh >> j) & 1u) : ((qh >> (j + 16u)) & 1u);
    uint nib = lp < 16u ? (by & 0xFu) : (by >> 4u);
    code = int(nib | (hb << 4u));
    wsc = d; wmn = 16.0 * d;
#elif GGUF_TYPE == 41                               // Q1_0
    uint base = rb + (e >> 7u) * 18u;
    float d = rdf16(base);
    uint b = e & 127u;
    uint bit = (rd8(base + 2u + (b >> 3u)) >> (b & 7u)) & 1u;
    code = bit > 0u ? 1 : -1;
    wsc = d;
#elif GGUF_TYPE == 12                               // Q4_K
    uint s = e >> 8u;
    uint base = rb + s * 144u;
    float d = rdf16(base), dmin = rdf16(base + 2u);
    uint se = e & 255u;
    uint chunk = se >> 6u, l = se & 31u, si = se >> 5u;
    uint by = rd8(base + 16u + chunk * 32u + l);
    uint nib = (se & 32u) == 0u ? (by & 0xFu) : (by >> 4u);
    float sc, mn; k4sm(si, base + 4u, sc, mn);
    code = int(nib);
    wsc = d * sc; wmn = dmin * mn;
#elif GGUF_TYPE == 13                               // Q5_K
    uint s = e >> 8u;
    uint base = rb + s * 176u;
    float d = rdf16(base), dmin = rdf16(base + 2u);
    uint se = e & 255u;
    uint chunk = se >> 6u, l = se & 31u, si = se >> 5u;
    uint by = rd8(base + 48u + chunk * 32u + l);
    uint u1 = 1u << (2u * chunk), u2 = 2u << (2u * chunk);
    uint nib = (se & 32u) == 0u ? (by & 0xFu) : (by >> 4u);
    uint hb = (se & 32u) == 0u ? ((rd8(base + 16u + l) & u1) != 0u ? 16u : 0u)
                               : ((rd8(base + 16u + l) & u2) != 0u ? 16u : 0u);
    float sc, mn; k4sm(si, base + 4u, sc, mn);
    code = int(nib | hb);
    wsc = d * sc; wmn = dmin * mn;
#elif GGUF_TYPE == 14                               // Q6_K
    uint base = rb + (e >> 8u) * 210u;
    float d = rdf16(base + 208u);
    uint se = e & 255u;
    uint npos = (se >> 7u) << 7u, r = se & 127u;
    uint l = r & 31u, which = r >> 5u, is_ = l >> 4u;
    uint ql0 = rd8(base + (npos >> 1u) + l);
    uint qh0 = rd8(base + 128u + (npos >> 2u) + l);
    uint sbi = (npos >> 7u) * 8u + is_;
    if (which == 0u) { code = int((ql0 & 0xFu) | (((qh0 >> 0u) & 3u) << 4u)) - 32; wsc = d * scf8(base + 192u + sbi + 0u); }
    else if (which == 1u) { code = int((rd8(base + (npos >> 1u) + l + 32u) & 0xFu) | (((qh0 >> 2u) & 3u) << 4u)) - 32; wsc = d * scf8(base + 192u + sbi + 2u); }
    else if (which == 2u) { code = int((ql0 >> 4u) | (((qh0 >> 4u) & 3u) << 4u)) - 32; wsc = d * scf8(base + 192u + sbi + 4u); }
    else { code = int((rd8(base + (npos >> 1u) + l + 32u) >> 4u) | (((qh0 >> 6u) & 3u) << 4u)) - 32; wsc = d * scf8(base + 192u + sbi + 6u); }
#elif GGUF_TYPE == 10                               // Q2_K
    uint base = rb + (e >> 8u) * 84u;
    float d = rdf16(base + 80u), dmin = rdf16(base + 82u);
    uint se = e & 255u;
    uint sc = rd8(base + (se >> 4u));
    uint byteIdx = (se >> 7u) * 32u + (se & 31u);
    uint shift = 2u * ((se & 127u) >> 5u);
    uint code2 = (rd8(base + 16u + byteIdx) >> shift) & 3u;
    code = int(code2);
    wsc = d * float(sc & 0xFu); wmn = dmin * float(sc >> 4u);
#elif GGUF_TYPE == 11                               // Q3_K
    uint base = rb + (e >> 8u) * 110u;
    float d = rdf16(base + 108u);
    uint se = e & 255u;
    int scc = q3s(se >> 4u, base + 96u);
    float dl = d * float(scc - 32);
    uint m = 1u << (se >> 5u);
    uint byteIdx = (se >> 7u) * 32u + (se & 31u);
    uint shift = 2u * ((se & 127u) >> 5u);
    uint code2 = (rd8(base + 32u + byteIdx) >> shift) & 3u;
    int hm = (rd8(base + (se & 31u)) & m) != 0u ? 0 : 4;
    code = int(code2) - hm;
    wsc = dl;
#endif
}
