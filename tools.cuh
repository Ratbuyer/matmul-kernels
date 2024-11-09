#include <stdio.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <random>
#include <map>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                              \
  {                                                                                      \
    cudaError_t error_code = callstr;                                                    \
    if (error_code != cudaSuccess)                                                       \
    {                                                                                    \
      std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
      assert(0);                                                                         \
    }                                                                                    \
  }
#endif

half rand_half()
{
  return __float2half(5.0f * rand() / RAND_MAX);
}

int rand_int(int max)
{
  return rand() % max;
}

void print_matrix(half *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%f ", __half2float(matrix[i * cols + j]));
    }
    printf("\n");
  }
  printf("\n");
}

void print_matrix(int *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%d ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void fill_random(half *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      float value = 1.0f * rand() / RAND_MAX;
      matrix[i * cols + j] = __float2half(value);
    }
  }
}

void fill_fixed(half *matrix, int rows, int cols, float value)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      matrix[i * cols + j] = __float2half(value);
    }
  }
}

void fill_tile(half *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      if (i / 8 == 0 && j / 8 == 0)
      {
        matrix[i * cols + j] = __float2half(1.0f);
      }
      else
      {
        matrix[i * cols + j] = __float2half(0.0f);
      }
    }
  }
}

void transpose(half *matrix, int rows, int cols) {
    // Create a temporary matrix to store the result
    half* temp = new half[rows * cols];

    // Transpose the matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Swap element at (i, j) to (j, i)
            temp[j * rows + i] = matrix[i * cols + j];
        }
    }

    // Copy the transposed matrix back to the original matrix
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = temp[i];
    }

    // Free the temporary matrix
    delete[] temp;
}


// element in each subtile has the same value,
// which is their tile number in row major order
void fill_tilewise(int *matrix, int rows, int cols, int tile_size_row, int tile_size_col)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      matrix[i * cols + j] = (i / tile_size_row) * (cols / tile_size_col) + j / tile_size_col;
    }
  }
}

void CPU_gemm(half *A, half *B, half *C, int M, int N, int K)
{
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      C[i * N + j] = 0;
      for (int k = 0; k < K; k++)
      {
        float a = __half2float(A[i * K + k]);
        float b = __half2float(B[k * N + j]);
        float c = __half2float(C[i * N + j]);
        float new_c = a * b + c;
        C[i * N + j] = __float2half(new_c);
      }
    }
  }
}

void compare_matrices(half *A, half *B, int rows, int cols)
{
  float total_diff = 0.0;
  int total_elements = rows * cols;

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      float a = __half2float(A[i * cols + j]);
      float b = __half2float(B[i * cols + j]);
      total_diff += fabs((a - b) / a);
    }
  }

  float percentage_diff = (total_diff / total_elements) * 100;
  printf("Total error: %.2f%%\n", percentage_diff);
}

void print_differnce(half *A, half *B, int rows, int cols, float tolerance)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      float a = __half2float(A[i * cols + j]);
      float b = __half2float(B[i * cols + j]);
      bool is_same = a - tolerance < b && a + tolerance > b;
      if (!is_same)
      {
        printf("Error at (%d, %d) : %f != %f\n", i, j, a, b);
      }
    }
  }
}

void compress24(half *dense, half *sparse, int rows, int cols)
{
  assert(rows * cols % 4 == 0);

  memset(sparse, 0, rows * cols / 2 * sizeof(half));

  int counter;

  for (int i = 0; i < rows * cols; i += 4)
  {
    int sparse_offset = i / 2;

    counter = 0;

    for (int j = 0; j < 4; j++)
    {
      float value = __half2float(dense[i + j]);
      if (value != 0)
      {
        assert(counter < 2);
        sparse[sparse_offset + counter] = dense[i + j];
        counter++;
      }
    }
  }
}

void fill_24(half *matrix, int rows, int cols)
{
  assert(rows * cols % 4 == 0);

  for (int i = 0; i < rows * cols; i += 4)
  {
    matrix[i] = 0.0;
    matrix[i + 1] = 0.0;
    matrix[i + 2] = 0.0;
    matrix[i + 3] = 0.0;

    int position1 = rand() % 4;
    int position2 = rand() % 4;

    // position2 = position2 == position1 ? (position2 + 1) % 4 : position2;

    // matrix[i + position1] = __float2half(1.0f);
    // matrix[i + position2] = __float2half(1.0f);

    matrix[i + position1] = __float2half(rand_half());
    matrix[i + position2] = __float2half(rand_half());
  }
}

__device__ __forceinline__ void
MMA_FP16_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__* a, uint32_t __restrict__* b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{ %0, %1 },"
                 "{ %2, %3, %4, %5 },"
                 "{ %6, %7 },"
                 "{ %8, %9 };"
                 : "=r"(c[0]), "=r"(c[1])    // Output operands
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),  // Input operands
                   "r"(b[0]), "r"(b[1]), 
                   "r"(c[0]), "r"(c[1])      // Input operands (initial values of c)
                 : "memory");                // Clobber list
}

template<int SizeInBytes>
__device__ __forceinline__ void cp_async(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

__device__ __forceinline__ void cp_async_group_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}