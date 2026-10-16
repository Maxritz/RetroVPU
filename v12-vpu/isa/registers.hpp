#pragma once
#include "types.hpp"
#include <string>

namespace v12 {

struct VirtualReg {
    std::string name;
    DType       dtype;
    bool        is_temp = true;   // temp vs input/output
};

inline VirtualReg reg_f32(const std::string& n) { return {n, DType::Float32, false}; }
inline VirtualReg reg_vec4(const std::string& n) { return {n, DType::Vec4F32, false}; }
inline VirtualReg reg_i8(const std::string& n)   { return {n, DType::Int8, false}; }
inline VirtualReg reg_i32(const std::string& n)   { return {n, DType::Int32, false}; }

} // namespace v12
