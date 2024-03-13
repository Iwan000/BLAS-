#include <benchmark/benchmark.h>
#include <iostream>
#include <cblas.h>
#include <arm_neon.h>
#include <cmath>


#ifndef OPENBLAS_CONST
# define OPENBLAS_CONST const
#endif

// typedef const OPENBLAS_CONST;
typedef int blasint;

void generateRandomMatrix(double* matrix, int rows, int cols) {
    srand(time(0));  // 使用当前时间作为随机数种子

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = static_cast<double>(rand()) / RAND_MAX;  // 生成0到1之间的随机数
        }
    }
}

void printMatrix(const double* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}


bool compareMatrices(const double* matrix1, const double* matrix2, int rows, int columns, int ld)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            if (std::fabs(matrix1[i * ld + j] - matrix2[i * ld + j]) > 0.000001)
            {
                return false;
            }
        }
    }
    return true;
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


double* arrangeMatrixBlock(const double* matrixA, int rows, int cols, int BLOCK_SIZE) {
    int m = rows; // 矩阵A的行数
    int n = cols; // 矩阵A的列数

    double* arrangedMatrix = new double[m * n];

    for (int i = 0; i < m; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int row = i; row < i + BLOCK_SIZE; row++) {
                for (int col = j; col < j + BLOCK_SIZE; col++) {
                    // 按照行主序排布子块元素并存储到一维数组中
                    int blockRow = (row - i) / BLOCK_SIZE; // 子块行索引
                    int blockCol = (col - j) / BLOCK_SIZE; // 子块列索引
                    int index = blockRow * (n / BLOCK_SIZE) + blockCol; // 子块索引
                    int flatIndex = row * n + col; // 在一维数组中的索引位置
                    arrangedMatrix[flatIndex] = matrixA[index];
                }
            }
        }
    }

    return arrangedMatrix;
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

int main() {
    const int M = 4;
    const int N = 4;
    const int K = 4;

    const double A[M * K] = {
        1.0, 2.0, 3.0, 4.0,
        6.0, 7.0, 8.0, 9.0, 
        11.0, 12.0, 13.0, 14.0, 
        16.0, 17.0, 18.0, 19.0
    };

    const double B[K * N] = {
        1.0, 4.0, 7.0,8.0,
        2.0, 5.0, 8.0,9.0,
        3.0, 6.0, 9.0,10.0,
        10.0, 13.0, 11.0,11.0
    };

    double C[M * N] = { 0.0 };

    const double alpha = 1.0;  // Scaling factor for the multiplication
    const double beta = 0.0;   // Scaling factor for matrix C

    // Perform matrix multiplication using cblas_dgemmOptimized_update
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, M, B, K, beta, C, M);

    // Print the result matrix
    printf("Result using cblas_dgemmOptimized_update:\n");
    printMatrix(C, M, N);

    double D[M * N] = { 0.0 };

    // Perform matrix multiplication using cblas_dgemm
    cblas_dgemmOptimized_update5(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, M, B, K, beta, D, M);

    // Print the result matrix
    printf("Result using cblas_dgemm:\n");
    printMatrix(D, M, N);

    printf("Comparison result: %d\n", compareMatrices(C, D, M, N, N));

    return 0;
}






