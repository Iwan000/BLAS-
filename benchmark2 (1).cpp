#include <benchmark/benchmark.h>
#include <iostream>
#include <cblas.h>
#include <arm_neon.h>
#include <cmath>

#include <fstream>
#include <sstream>
#include <vector>
#include <omp.h>


void generateRandomMatrix(double* matrix, int rows, int cols) {
    srand(time(0));  // 使用当前时间作为随机数种子

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = static_cast<double>(rand()) / RAND_MAX;  // 生成0到1之间的随机数
        }
    }
}



#ifndef OPENBLAS_CONST
# define OPENBLAS_CONST const
#endif

// typedef const OPENBLAS_CONST;
typedef int blasint;

void matrixMultiplication(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                          OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                          OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double* A,
                          OPENBLAS_CONST blasint lda, OPENBLAS_CONST double* B, OPENBLAS_CONST blasint ldb,
                          OPENBLAS_CONST double beta, double* C, OPENBLAS_CONST blasint ldc) {
    for (blasint i = 0; i < M; ++i) {
        for (blasint j = 0; j < N; ++j) {
            double sum = 0.0;
            for (blasint k = 0; k < K; ++k) {
                double a = (TransA == CblasTrans) ? A[k * lda + i] : A[i * lda + k];
                double b = (TransB == CblasTrans) ? B[j * ldb + k] : B[k * ldb + j];
                sum += a * b;
            }
            C[j * ldc + i] = beta * C[j * ldc + i] + alpha * sum;
        }
    }
}

void transposeMatrix(OPENBLAS_CONST double* src, double* dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}
void transposeMatrix2(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = i + 1; j < cols; j++) {
            // Swap the elements at (i, j) and (j, i)
            double temp = matrix[i * cols + j];
            matrix[i * cols + j] = matrix[j * rows + i];
            matrix[j * rows + i] = temp;
        }
    }
}

void cblas_dgemmOptimized_update5(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                                 OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                                 OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double* A,
                                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST double* B, OPENBLAS_CONST blasint ldb,
                                 OPENBLAS_CONST double beta, double* C, OPENBLAS_CONST blasint ldc)
{
    if(TransA == CblasTrans){
        double* transposedA = (double*)malloc(M * K * sizeof(double));
        transposeMatrix(A, transposedA, K, M);
        A = transposedA;
    }
    if(TransB == CblasTrans){
        double* transposedB = (double*)malloc(K * N * sizeof(double));
        transposeMatrix(B, transposedB, N ,K);
        B = transposedB;
    }

    const int BLOCK_SIZE = 128; // Set the block size

    
    if (Order == CblasRowMajor)
    {
        #pragma omp parallel for
        for (int i = 0; i < M; i += BLOCK_SIZE)
        {
            for (int j = 0; j < N; j += BLOCK_SIZE)
            {
                for (int k = 0; k < K; k += BLOCK_SIZE)
                {
                    for (int ii = i; ii < i + BLOCK_SIZE && ii < M; ii += 4)
                    {
                        for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj += 4)
                        {
                            
                            // Registers
                            double sum[4][4] = {{0.0}};
                            for (int kk = k; kk < k + BLOCK_SIZE && kk < K; ++kk)
                            {
                                // Load values from memory into registers
                                double a0 = A[( ii) * lda + ( kk)];
                                double a1 = A[( ii + 1) * lda + ( kk)];
                                double a2 = A[( ii + 2) * lda + ( kk)];
                                double a3 = A[( ii + 3) * lda + ( kk)];

                                for (int x = 0; x < 4; ++x)
                                {
                                    double b0 = B[( kk) * ldb + ( jj + x)];

                                    sum[x][0] += a0 * b0;
                                    sum[x][1] += a1 * b0;
                                    sum[x][2] += a2 * b0;
                                    sum[x][3] += a3 * b0;
                                }
                            }

                            // Update C directly from the registers
                            for (int k = 0; k < 4; ++k) {
                                for (int l = 0; l < 4; ++l) {
                                    C[(ii + l) * ldc + (jj + k)] = beta * C[(ii + l) * ldc + (jj + k)] + alpha * sum[k][l];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else if (Order == CblasColMajor)
    {
         double* transposedA = (double*)malloc(M * K * sizeof(double));
         double* transposedB = (double*)malloc(K * N * sizeof(double));
         transposeMatrix(A, transposedA, K, M);
         A = transposedA;
         transposeMatrix(B, transposedB, N ,K);
         B = transposedB;

        #pragma omp parallel for
        for (int j = 0; j < N; j += BLOCK_SIZE)
        {
            for (int i = 0; i < M; i += BLOCK_SIZE)
            {
                for (int k = 0; k < K; k += BLOCK_SIZE)
                {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj += 4)
                    {
                        for (int ii = i; ii < i + BLOCK_SIZE && ii < M; ii += 4)
                        {
                            
                            // Registers
                            double sum[4][4] = {{0.0}};

                            for (int kk = k; kk < k + BLOCK_SIZE && kk < K; ++kk)
                            {
                                // Load values from memory into registers
                                double a0 = A[( ii) * lda + ( kk)];
                                double a1 = A[( ii + 1) * lda + (kk)];
                                double a2 = A[( ii + 2) * lda + ( kk)];
                                double a3 = A[( ii + 3) * lda + (kk)];

                                for (int x = 0; x < 4; ++x)
                                {
                                    double b0 = B[( kk) * ldb + ( jj + x)];

                                    sum[x][0] += a0 * b0;
                                    sum[x][1] += a1 * b0;
                                    sum[x][2] += a2 * b0;
                                    sum[x][3] += a3 * b0;
                                }
                            }

                            // Update C from the registers
                            for (int k = 0; k < 4; ++k) {
                                for (int l = 0; l < 4; ++l) {
                                    C[(ii + l) * ldc + (jj + k)] = beta * C[(ii + l) * ldc + (jj + k)] + alpha * sum[k][l];
                                }
                            }
                        }
                    }
                }
            }
        }
        transposeMatrix2(C, M ,N);
    }
}

// Define the function to benchmark
static void BM_cblas_dgemmOptimized_update(benchmark::State& state) {
    int size = state.range(0);  // Get the current data size from the benchmark state

    // Create matrices
    double* A = new double[size * size];
    double* B = new double[size * size];
    double* C = new double[size * size];

    // Generate random matrices
    generateRandomMatrix(A, size, size);
    generateRandomMatrix(B, size, size);

    // Run the benchmark loop
    for (auto _ : state) {
        cblas_dgemmOptimized_update5(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1.0, A, size, B, size, 0.0, C, size); // Call your matrix multiplication function
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1.0, A, size, B, size, 0.0, C, size);
        benchmark::DoNotOptimize(C);  // Prevent the compiler from optimizing the result
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
}

// Main function
int main(int argc, char** argv) {
    // Initialize the benchmark
    ::benchmark::Initialize(&argc, argv);

    // Run the benchmark for different data sizes
    ::benchmark::RegisterBenchmark("cblas_dgemmOptimized_update", BM_cblas_dgemmOptimized_update)
        ->Arg(100)  // Add the data sizes you want to test
        ->Arg(500)
        ->Arg(1000)
        ->Arg(2000)
        ->Arg(3000)
        ->Arg(4000)
        ->Arg(5000)
        ->Arg(6000)
        ->Arg(7000)
        ->Arg(8000)
        
        ->Unit(benchmark::kMillisecond);

    // Plot the results
    ::benchmark::RunSpecifiedBenchmarks();
    
    return 0;
}


