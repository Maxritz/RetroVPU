#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>

namespace VPU {
    namespace Level2Ref {
        void sgemv_ref(bool trans, size_t M, size_t N, float alpha,
                       const float* A, size_t lda,
                       const float* x, size_t incx,
                       float beta, float* y, size_t incy) {
            if (!trans) {
                for (size_t i = 0; i < M; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < N; ++j) {
                        sum += A[j * lda + i] * x[j * incx];
                    }
                    y[i * incy] = alpha * sum + beta * y[i * incy];
                }
            }
        }
    }
}

int main() {
    std::cout << "Starting test..." << std::endl;
    size_t M = 512, N = 256;
    std::vector<float> A(M * N), x(N), y(M), y_copy(M);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < M * N; ++i) A[i] = dist(rng);
    for (size_t i = 0; i < N; ++i) x[i] = dist(rng);
    std::copy(y.begin(), y.end(), y_copy.begin());
    
    std::cout << "Calling sgemv_ref..." << std::endl;
    VPU::Level2Ref::sgemv_ref(false, M, N, 1.0f, A.data(), M, x.data(), 1, 0.0f, y.data(), 1);
    std::cout << "sgemv_ref done" << std::endl;
    
    return 0;
}
