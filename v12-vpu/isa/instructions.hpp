#pragma once
#include "types.hpp"
#include <string>
#include <vector>
#include <variant>

namespace v12 {

enum class Op : u32 {
    Invalid     = 0,
    Load        = 1,
    Store       = 2,
    Mov         = 3,
    Mat4        = 4,
    ScaleBias   = 5,
    TexLookup   = 6,
    DotF32      = 7,
    DotI8       = 8,
    MatVecQ8    = 9,
    ReduceSum   = 10,
};

inline const char* op_name(Op o) {
    switch (o) {
        case Op::Load:      return "LOAD";
        case Op::Store:     return "STORE";
        case Op::Mov:       return "MOV";
        case Op::Mat4:      return "MAT4";
        case Op::ScaleBias: return "SCALE_BIAS";
        case Op::TexLookup: return "TEX_LOOKUP";
        case Op::DotF32:    return "DOT_F32";
        case Op::DotI8:     return "DOT_I8";
        case Op::MatVecQ8:  return "MATVEC_Q8";
        case Op::ReduceSum: return "REDUCE_SUM";
        default:            return "INVALID";
    }
}

enum class Origin : u32 {
    VirtualISA,
    V12Inspired,
    NewAIExtension,
};

struct Instr {
    Op          op        = Op::Invalid;
    std::string dst;          // destination virtual register
    std::vector<std::string> src;   // source virtual registers
    std::vector<f32>       imm;     // immediate values (float, bit-reinterpreted)
    std::vector<i32>       immI;    // immediate values (int32)
    DType       dtype     = DType::Float32;
    Origin      origin    = Origin::VirtualISA;
    std::string note;         // documentation only

    // Accessors for common patterns
    bool is_pure() const {
        switch (op) {
            case Op::Load: case Op::Store: case Op::Mov:
                return op == Op::Mov || op == Op::Load;
            default:
                return true; // pure if no side effects
        }
    }
};

struct Program {
    std::vector<Instr> instrs;
    std::string        output;   // final destination register name
    std::string        name;
};

} // namespace v12
