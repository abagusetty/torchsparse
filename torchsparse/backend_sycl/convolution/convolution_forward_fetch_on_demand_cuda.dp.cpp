/*
Please consider citing the following paper when using the code:

@inproceedings{hong2023pcengine,
  title={{Exploiting Hardware Utilization and Adaptive Dataflow for Efficient Sparse Convolution in 3D Point Clouds}},
  author={Hong, Ke and Yu, Zhongming and Dai, Guohao and Yang, Xinhao and Lian, Yaoxiu and Liu, Zehao and Xu, Ningyi and Wang, Yu},
  booktitle={Sixth Conference on Machine Learning and Systems (MLSys)},
  year={2023}
}
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <dpct/blas_utils.hpp>

#include <torch/extension.h>
#if DPCT_COMPATIBILITY_TEMP >= 700
#endif

#include "convolution_forward_fetch_on_demand_cuda.h"
#include <sycl/ext/intel/math.hpp>

#include <cmath>

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)

// kernels employed in PCEngine [Fetch-on-Demand]
// device function to indicate the weight index in fetch-on-demand gemms
__dpct_inline__ int binary_search(const int *S_csrRowPtr, const int eid,
                                  const int start, const int end) {

  int lo = start, hi = end;
  if (lo == hi){
    return lo;
  }
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    /*
    DPCT1098:521: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    if (*(S_csrRowPtr + mid) <= eid) {
        lo = mid + 1;
    } else {
        hi = mid;
    }
  }
  /*
  DPCT1098:522: The '*' expression is used instead of the __ldg call. These two
  expressions do not provide the exact same functionality. Check the generated
  code for potential precision and/or performance issues.
  */
  if (*(S_csrRowPtr + hi) <= eid) {
    return hi;
  } else {
      return hi - 1;
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
void fetch_on_demand_gemm_fp32(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y =
      BLOCK_SIZE * N_LOOP * by +
      ty
      /*
      DPCT1098:523: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      - qkpos[widx] + kpos[widx];

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((sycl::float4 *)(&Bs[ty][ctx])) =
        ((s + ty) < c_in && cx < c_out)
            ? *((sycl::float4 *)(kw_ptr + c_out * (s + ty) + cx))
            : *((sycl::float4 *)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      /*
      DPCT1098:524: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      int in_row = y_temp < kpos[widx + 1] ? imap[y_temp] : -1;

      *((sycl::float4 *)(&As[n][ty][ctx])) =
          ((s + ctx) < c_in && in_row > -1)
              ? *((sycl::float4 *)(&in_f[c_in * in_row + s + ctx]))
              : *((sycl::float4 *)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:483: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:484: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    /*
    DPCT1098:525: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int out_row = y_temp < kpos[widx + 1] ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
void fetch_on_demand_gemm_no_fusion_fp32(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 2;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  // const float *kw_ptr = &kw[widx * c_in * c_out];
  const float *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((sycl::float4 *)(&Bs[ty][ctx])) =
        ((s + ty) < c_in && cx < c_out)
            ? *((sycl::float4 *)(kw_ptr + c_out * (s + ty) + cx))
            : *((sycl::float4 *)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((sycl::float4 *)(&As[n][ty][ctx])) =
          ((s + ctx) < c_in && in_row > -1)
              ? *((sycl::float4 *)(&in_f[c_in * in_row + s + ctx]))
              : *((sycl::float4 *)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:485: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:486: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        // atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
        out_f[c_out * out_row + cx + c] += Csub[n][c];
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, blockDim.x = 4, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
void fetch_on_demand_gemm_fp32_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y =
      BLOCK_SIZE * N_LOOP * by +
      ty
      /*
      DPCT1098:526: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      - qkpos[widx] + kpos[widx];

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

  // Load the matrices from device memory
  // to shared memory; each thread loads
  // one element of each matrix

  // Kernel weight to Bs
  *((sycl::float4 *)(&Bs[ty][ctx])) =
      ((ty) < c_in && cx < c_out)
          ? *((sycl::float4 *)(kw_ptr + c_out * (ty) + cx))
          : *((sycl::float4 *)(&padding[0]));

  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    int y_temp = y + n * BLOCK_SIZE;

    // The thread deals with the x-th channel of the y-th output
    /*
    DPCT1098:527: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int in_row = y_temp < kpos[widx + 1] ? imap[y_temp] : -1;

    *((sycl::float4 *)(&As[n][ty][ctx])) =
        ((ctx) < c_in && in_row > -1)
            ? *((sycl::float4 *)(&in_f[c_in * in_row + ctx]))
            : *((sycl::float4 *)(&padding[0]));
  }

  // Synchronize to make sure the matrices are loaded
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Multiply the two matrices together;
  // each thread computes one element
  // of the block sub-matrix
#pragma unroll 
  for (int n = 0; n < N_LOOP; n++){
#pragma unroll
    for (int k = 0; k < c_in; ++k) {
      float Ast = As[n][ty][k];
#pragma unroll
      for (int c = 0; c < 4; c++){
        Csub[n][c] += Ast * Bs[k][ctx + c];
      }
    }
    int y_temp = y + n * BLOCK_SIZE;
    /*
    DPCT1098:528: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int out_row = y_temp < kpos[widx + 1] ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}

/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, 
blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
void fetch_on_demand_gemm_fp32_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y =
      BLOCK_SIZE * N_LOOP * by +
      ty
      /*
      DPCT1098:529: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      - qkpos[widx] + kpos[widx];

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][2] = {0.0f};
  float padding[2] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((sycl::float2 *)(&Bs[ty][ctx])) =
        ((s + ty) < c_in && cx < c_out)
            ? *((sycl::float2 *)(kw_ptr + c_out * (s + ty) + cx))
            : *((sycl::float2 *)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      /*
      DPCT1098:530: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      int in_row = y_temp < kpos[widx + 1] ? imap[y_temp] : -1;

      *((sycl::float2 *)(&As[n][ty][ctx])) =
          ((s + ctx) < c_in && in_row > -1)
              ? *((sycl::float2 *)(&in_f[c_in * in_row + s + ctx]))
              : *((sycl::float2 *)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:487: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:488: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    /*
    DPCT1098:531: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int out_row = y_temp < kpos[widx + 1] ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, 
blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
void fetch_on_demand_gemm_fp32_1(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  // const int ctx = tx;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y =
      BLOCK_SIZE * N_LOOP * by +
      ty
      /*
      DPCT1098:532: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      - qkpos[widx] + kpos[widx];

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP] = {0.0f};
  float padding = 0.0f;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      /*
      DPCT1098:533: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      int in_row = y_temp < kpos[widx + 1] ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:489: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // float Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
        Csub[n] += As[n][ty][k] * Bs[k][tx];
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:490: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    /*
    DPCT1098:534: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int out_row = y_temp < kpos[widx + 1] ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      // }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, 
blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
void fetch_on_demand_gemm_no_fusion_fp32_1(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  // const int ctx = tx;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP] = {0.0f};
  float padding = 0.0f;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:491: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // float Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
        Csub[n] += As[n][ty][k] * Bs[k][tx];
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:492: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      // atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      out_f[c_out * out_row + cx] += Csub[n];
      // }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
void fetch_on_demand_gemm_fp16_4_once(
    const int *__restrict__ kpos, const int *__restrict__ qkpos,
    const int k_vol, const int c_in, const int c_out,
    const sycl::half *__restrict__ in_f, const sycl::half *__restrict__ kw,
    sycl::half *out_f, const int *imap, const int *omap) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const sycl::half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y =
      BLOCK_SIZE * N_LOOP * by +
      ty
      /*
      DPCT1098:535: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      - qkpos[widx] + kpos[widx];

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  sycl::half Csub[N_LOOP][4] = {sycl::ext::intel::math::float2half_rn(0.0f)};
  sycl::half padding[4] = {sycl::ext::intel::math::float2half_rn(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

  // Kernel weight to Bs
  *((sycl::float2 *)(&Bs[ty][ctx])) =
      (ty < c_in && cx < c_out) ? *((sycl::float2 *)(kw_ptr + c_out * ty + cx))
                                : *((sycl::float2 *)(&padding[0]));

  int y_temp = y;
  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    // The thread deals with the x-th channel of the y-th output
    /*
    DPCT1098:536: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int in_row = y_temp < kpos[widx + 1] ? imap[y_temp] : -1;

    *((sycl::float2 *)(&As[n][ty][ctx])) =
        (ctx < c_in && in_row > -1)
            ? *((sycl::float2 *)(&in_f[c_in * in_row + ctx]))
            : *((sycl::float2 *)(&padding[0]));

    y_temp += BLOCK_SIZE;
  }

  // Synchronize to make sure the matrices are loaded
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Multiply the two matrices together;
  // each thread computes one element
  // of the block sub-matrix
#pragma unroll 
  for (int n = 0; n < N_LOOP; n++){
#pragma unroll
    for (int k = 0; k < c_in; ++k){
      sycl::half Ast = As[n][ty][k];
#pragma unroll
      for (int c = 0; c < 4; c++){
        Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
      }
    }

    int y_temp = y + n * BLOCK_SIZE;
    /*
    DPCT1098:537: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int out_row = y_temp < kpos[widx + 1] ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
#else
  #pragma message("FP16 kernels will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
void fetch_on_demand_gemm_fp16_2(const int *__restrict__ kpos,
                                 const int *__restrict__ qkpos, const int k_vol,
                                 const int c_in, const int c_out,
                                 const sycl::half *__restrict__ in_f,
                                 const sycl::half *__restrict__ kw,
                                 sycl::half *out_f, const int *imap,
                                 const int *omap) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const sycl::half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y =
      BLOCK_SIZE * N_LOOP * by +
      ty
      /*
      DPCT1098:538: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      - qkpos[widx] + kpos[widx];

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  sycl::half Csub[N_LOOP][2] = {sycl::ext::intel::math::float2half_rn(0.0f)};
  sycl::half padding[2] = {sycl::ext::intel::math::float2half_rn(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      /*
      DPCT1098:539: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      int in_row = y_temp < kpos[widx + 1] ? imap[y_temp] : -1;

      *((float*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:493: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        sycl::half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:494: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    /*
    DPCT1098:540: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int out_row = y_temp < kpos[widx + 1] ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
void fetch_on_demand_gemm_fp16_1(const int *__restrict__ kpos,
                                 const int *__restrict__ qkpos, const int k_vol,
                                 const int c_in, const int c_out,
                                 const sycl::half *__restrict__ in_f,
                                 const sycl::half *__restrict__ kw,
                                 sycl::half *out_f, const int *imap,
                                 const int *omap) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  // const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const sycl::half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y =
      BLOCK_SIZE * N_LOOP * by +
      ty
      /*
      DPCT1098:541: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      - qkpos[widx] + kpos[widx];

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  sycl::half Csub[N_LOOP] = {sycl::ext::intel::math::float2half_rn(0.0f)};
  sycl::half padding = sycl::ext::intel::math::float2half_rn(0.0f);

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      /*
      DPCT1098:542: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      int in_row = y_temp < kpos[widx + 1] ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:495: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // half Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
          Csub[n] = __hfma(As[n][ty][k], Bs[k][tx], Csub[n]);
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:496: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    /*
    DPCT1098:543: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int out_row = y_temp < kpos[widx + 1] ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      // }
    }
  }
#else
  #pragma message("FP16 kernels will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
void fetch_on_demand_gemm_no_fusion_fp16_1(const int knnz, const int c_in,
                                           const int c_out,
                                           const sycl::half *__restrict__ in_f,
                                           const sycl::half *__restrict__ kw,
                                           sycl::half *out_f, const int *imap,
                                           const int *omap) {

#if DPCT_COMPATIBILITY_TEMP >= 700

  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  // const int ctx = tx << 1;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const sycl::half *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  sycl::half Csub[N_LOOP] = {sycl::ext::intel::math::float2half_rn(0.0f)};
  sycl::half padding = sycl::ext::intel::math::float2half_rn(0.0f);

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:497: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:544: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // half Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
          Csub[n] = __hfma(As[n][ty][k], Bs[k][tx], Csub[n]);
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:498: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:545: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      // atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      out_f[c_out * out_row + cx] = 
        __hadd(out_f[c_out * out_row + cx], Csub[n]);
      // }
    }
  }
#else
  #pragma message("FP16 kernels will not be compiled.")
#endif
}


// kernels using tensor cores
// using namespace nvcuda;
/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, int M, int K, int N, int WS,
          int MS, int NS>
/*
DPCT1110:499: The total declared local variable size in device function
fetch_on_demand_gemm_tf32 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void fetch_on_demand_gemm_tf32(const int *__restrict__ kpos,
                               const int *__restrict__ qkpos, const int k_vol,
                               const int c_in, const int c_out,
                               const float *__restrict__ in_f,
                               const float *__restrict__ kw, float *out_f,
                               const int *imap, const int *omap) {
#if DPCT_COMPATIBILITY_TEMP >= 800

  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 2;
  const int tid = ty * item_ct1.get_local_range(2) + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y =
      BLOCK_SIZE * N_LOOP * by +
      ty
      /*
      DPCT1098:548: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      - qkpos[widx] + kpos[widx];

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::fill_fragment(c[n], 0.0f);
  }
  
  // May not be necessary
  /*
  DPCT1065:546: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((sycl::float4 *)(&Bs[ty][ctx])) =
        ((s + ty) < c_in && cx < c_out)
            ? *((sycl::float4 *)(kw_ptr + c_out * (s + ty) + cx))
            : *((sycl::float4 *)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      /*
      DPCT1098:551: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      int in_row = y_temp < kpos[widx + 1] ? imap[y_temp] : -1;

      *((sycl::float4 *)(&As[n][ty][ctx])) =
          ((s + ctx) < c_in && in_row > -1)
              ? *((sycl::float4 *)(&in_f[c_in * in_row + s + ctx]))
              : *((sycl::float4 *)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:500: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:549: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      dpct::experimental::matrix::joint_matrix<
          dpct::experimental::matrix::a, M, N, K, nvcuda::wmma::precision::tf32,
          dpct::experimental::matrix::row_major>
          a[N_LOOP / 2];
      dpct::experimental::matrix::joint_matrix<
          dpct::experimental::matrix::b, M, N, K, nvcuda::wmma::precision::tf32,
          dpct::experimental::matrix::row_major>
          b;
      nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = nvcuda::wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = nvcuda::wmma::__float_to_tf32(a[n].x[t]);
        }
        nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:501: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:550: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::store_matrix_sync(
        &As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], c[n],
        BLOCK_SIZE + SKEW,
        sycl::ext::oneapi::experimental::matrix::layout::row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  /*
  DPCT1065:547: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    /*
    DPCT1098:552: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int out_row = y_temp < kpos[widx + 1] ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
void fetch_on_demand_gemm_no_fusion_tf32(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {
#if DPCT_COMPATIBILITY_TEMP >= 800
  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 2;
  const int tid = ty * item_ct1.get_local_range(2) + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  // const float *kw_ptr = &kw[widx * c_in * c_out];
  const float *kw_ptr = &kw[0];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::fill_fragment(c[n], 0.0f);
  }
  
  // May not be necessary
  /*
  DPCT1065:553: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((sycl::float4 *)(&Bs[ty][ctx])) =
        ((s + ty) < c_in && cx < c_out)
            ? *((sycl::float4 *)(kw_ptr + c_out * (s + ty) + cx))
            : *((sycl::float4 *)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((sycl::float4 *)(&As[n][ty][ctx])) =
          ((s + ctx) < c_in && in_row > -1)
              ? *((sycl::float4 *)(&in_f[c_in * in_row + s + ctx]))
              : *((sycl::float4 *)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:502: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:555: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      dpct::experimental::matrix::joint_matrix<
          dpct::experimental::matrix::a, M, N, K, nvcuda::wmma::precision::tf32,
          dpct::experimental::matrix::row_major>
          a[N_LOOP / 2];
      dpct::experimental::matrix::joint_matrix<
          dpct::experimental::matrix::b, M, N, K, nvcuda::wmma::precision::tf32,
          dpct::experimental::matrix::row_major>
          b;
      nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = nvcuda::wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = nvcuda::wmma::__float_to_tf32(a[n].x[t]);
        }
        nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]); 
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:503: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:556: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::store_matrix_sync(
        &As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], c[n],
        BLOCK_SIZE + SKEW,
        sycl::ext::oneapi::experimental::matrix::layout::row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  /*
  DPCT1065:554: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        // atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
        out_f[c_out * out_row + cx + c] += As[n][ty][ctx + c];
      }
    }
  }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif
}
////////////////////////////// CUDA_ARCH >= 800 for TF32 ///////////////////////////////////

/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, int M, int K, int N, int WS,
          int MS, int NS>
void fetch_on_demand_gemm_fp16_tc4(
    const int *__restrict__ kpos, const int *__restrict__ qkpos,
    const int k_vol, const int c_in, const int c_out,
    const sycl::half *__restrict__ in_f, const sycl::half *__restrict__ kw,
    sycl::half *out_f, const int *imap, const int *omap) {
#if DPCT_COMPATIBILITY_TEMP >= 700

  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 2;
  const int tid = ty * item_ct1.get_local_range(2) + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const sycl::half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y =
      BLOCK_SIZE * N_LOOP * by +
      ty
      /*
      DPCT1098:557: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      - qkpos[widx] + kpos[widx];

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  sycl::half padding[4] = {sycl::ext::intel::math::float2half_rn(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::fill_fragment(c[n],
                                sycl::ext::intel::math::float2half_rn(0.0f));
  }
  
  // May not be necessary
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((sycl::float2 *)(&Bs[ty][ctx])) =
        ((s + ty) < c_in && cx < c_out)
            ? *((sycl::float2 *)(kw_ptr + c_out * (s + ty) + cx))
            : *((sycl::float2 *)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      /*
      DPCT1098:558: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      int in_row = y_temp < kpos[widx + 1] ? imap[y_temp] : -1;

      *((sycl::float2 *)(&As[n][ty][ctx])) =
          ((s + ctx) < c_in && in_row > -1)
              ? *((sycl::float2 *)(&in_f[c_in * in_row + s + ctx]))
              : *((sycl::float2 *)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:504: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      dpct::experimental::matrix::joint_matrix<
          dpct::experimental::matrix::a, M, N, K, sycl::half,
          dpct::experimental::matrix::row_major>
          a[N_LOOP / 2];
      dpct::experimental::matrix::joint_matrix<
          dpct::experimental::matrix::b, M, N, K, sycl::half,
          dpct::experimental::matrix::row_major>
          b;
      nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
        nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:505: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::store_matrix_sync(
        &As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], c[n],
        BLOCK_SIZE + SKEW,
        sycl::ext::oneapi::experimental::matrix::layout::row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    /*
    DPCT1098:559: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int out_row = y_temp < kpos[widx + 1] ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#else
  #pragma message("FP16 kernels will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, int M, int K, int N, int WS,
          int MS, int NS>
void fetch_on_demand_gemm_fp16_tc4_async(
    const int *__restrict__ kpos, const int *__restrict__ qkpos,
    const int k_vol, const int c_in, const int c_out,
    const sycl::half *__restrict__ in_f, const sycl::half *__restrict__ kw,
    sycl::half *out_f, const int *imap, const int *omap) {
#if DPCT_COMPATIBILITY_TEMP >= 700

  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 2;
  const int tid = ty * item_ct1.get_local_range(2) + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const sycl::half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y =
      BLOCK_SIZE * N_LOOP * by +
      ty
      /*
      DPCT1098:560: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      - qkpos[widx] + kpos[widx];

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  sycl::half padding[4] = {sycl::ext::intel::math::float2half_rn(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Pipelined copy between gmem and shmem
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape4 =
      cuda::aligned_size_t<alignof(float2)>(sizeof(sycl::float2));

  // Fragments to store As, Bs and Cs
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::fill_fragment(c[n],
                                sycl::ext::intel::math::float2half_rn(0.0f));
  }
  
  // May not be necessary
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    // const half *kw2Bs_ptr = ((s + ty) < c_in && cx < c_out) ? 
    //   kw_ptr + c_out * (s + ty) + cx : &padding[0];
    pipe.producer_acquire();
    if ((s + ty) < c_in && cx < c_out){
      cuda::memcpy_async(&Bs[ty][ctx], kw_ptr + c_out * (s + ty) + cx, shape4, pipe);
    }
    else{
      cuda::memcpy_async(&Bs[ty][ctx], &padding[0], shape4, pipe);
    }
    // cuda::memcpy_async(&Bs[ty][ctx], kw2Bs_ptr, shape4, pipe);
    pipe.producer_commit();
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      /*
      DPCT1098:561: The '*' expression is used instead of the __ldg call. These
      two expressions do not provide the exact same functionality. Check the
      generated code for potential precision and/or performance issues.
      */
      int in_row = y_temp < kpos[widx + 1] ? imap[y_temp] : -1;

      // const half *inf2As_ptr = ((s + ctx) < c_in && in_row > -1) ? 
      //   &in_f[c_in * in_row + s + ctx] : &padding[0];
      pipe.producer_acquire();
      if ((s + ctx) < c_in && in_row > -1){
        cuda::memcpy_async(&As[n][ty][ctx], &in_f[c_in * in_row + s + ctx], shape4, pipe);
      }
      else{
        cuda::memcpy_async(&As[n][ty][ctx], &padding[0], shape4, pipe);
      }
      // cuda::memcpy_async(&As[n][ty][ctx], inf2As_ptr, shape4, pipe);
      pipe.producer_commit();
    }

    // Synchronize to make sure the matrices are loaded
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    /*
    DPCT1118:506: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      dpct::experimental::matrix::joint_matrix<
          dpct::experimental::matrix::a, M, N, K, sycl::half,
          dpct::experimental::matrix::row_major>
          a[N_LOOP / 2];
      dpct::experimental::matrix::joint_matrix<
          dpct::experimental::matrix::b, M, N, K, sycl::half,
          dpct::experimental::matrix::row_major>
          b;
      nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
      }  
#pragma unroll 
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    pipe.consumer_release();
    /*
    DPCT1118:507: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::store_matrix_sync(
        &As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], c[n],
        BLOCK_SIZE + SKEW,
        sycl::ext::oneapi::experimental::matrix::layout::row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    /*
    DPCT1098:562: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    int out_row = y_temp < kpos[widx + 1] ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#else
  #pragma message("FP16 kernels with asynchronous copy will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, int M, int K, int N, int WS,
          int MS, int NS>
void fetch_on_demand_gemm_no_fusion_fp16(const int knnz, const int c_in,
                                         const int c_out,
                                         const sycl::half *__restrict__ in_f,
                                         const sycl::half *__restrict__ kw,
                                         sycl::half *out_f, const int *imap,
                                         const int *omap) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  // Block index
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int bx = item_ct1.get_group(2);
  const int by = item_ct1.get_group(1);

  // Thread index
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int ctx = tx << 2;
  const int tid = ty * item_ct1.get_local_range(2) + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  // const half *kw_ptr = &kw[widx * c_in * c_out];
  const sycl::half *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  sycl::half padding[4] = {sycl::ext::intel::math::float2half_rn(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  auto &As = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  auto &Bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
      sycl::half[BLOCK_SIZE][BLOCK_SIZE + SKEW]>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::fill_fragment(c[n],
                                sycl::ext::intel::math::float2half_rn(0.0f));
  }
  
  // May not be necessary
  /*
  DPCT1065:563: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((sycl::float2 *)(&Bs[ty][ctx])) =
        ((s + ty) < c_in && cx < c_out)
            ? *((sycl::float2 *)(kw_ptr + c_out * (s + ty) + cx))
            : *((sycl::float2 *)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((sycl::float2 *)(&As[n][ty][ctx])) =
          ((s + ctx) < c_in && in_row > -1)
              ? *((sycl::float2 *)(&in_f[c_in * in_row + s + ctx]))
              : *((sycl::float2 *)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1118:508: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:565: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      dpct::experimental::matrix::joint_matrix<
          dpct::experimental::matrix::a, M, N, K, sycl::half,
          dpct::experimental::matrix::row_major>
          a[N_LOOP / 2];
      dpct::experimental::matrix::joint_matrix<
          dpct::experimental::matrix::b, M, N, K, sycl::half,
          dpct::experimental::matrix::row_major>
          b;
      nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
        nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1118:509: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:566: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::store_matrix_sync(
        &As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], c[n],
        BLOCK_SIZE + SKEW,
        sycl::ext::oneapi::experimental::matrix::layout::row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  /*
  DPCT1065:564: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        // out_f[c_out * out_row + cx + c] += As[n][ty][ctx + c];
        out_f[c_out * out_row + cx + c] = 
          __hadd(out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#else
  #pragma message("FP16 kernels will not be compiled.")
#endif
}
///////////////////////////////// CUDA_ARCH >= 700 ///////////////////////////////////




// in_feat: (N, c) N=# of input points, c = input channels
// out_feat: (M, o) M=# of output points, o = output channels
//                  for stride=1, M=N. For stride>1, the N input coords
//                  are requantized to M points with grid size (stride *
//                  cur_stride)
// kernel: (k^3, c, o) for a 3D convolution of length k
// neighbor_map: (a, 2) the hash table query results from in_coords to
//                      out_coords
//                      where neighbor_map[:,0] is the index of the input
//                      feature and neighbor_map[:,1] is the index of the output
//                      feature
// neighbor_offset: (k^3) count of active weights based on neighbor_map
//                      with unused weights having 0 and neighbor_offset[k^3/2]
//                      holding w[0,0].
at::Tensor conv_forward_fetch_on_demand_cuda(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    const int sum_nnz, at::Tensor neighbor_address,
    at::Tensor q_neighbor_address, const int output_size, const int qsum_nnz,
    const bool transpose, const bool allow_tf32, const bool allow_fp16) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();

  // int sum_nnz = (int)torch::sum(neighbor_offset).item<int>();
  int input_size = in_feat.size(0);
  int in_channel = in_feat.size(1);
  int out_channel = kernel.size(2);
  int k_vol = kernel.size(0);
  // int *knnz_ptr = neighbor_offset.data_ptr<int>();
  // int *in_map_ptr = in_neighbor_map.data_ptr<int>();
  // int *out_map_ptr = out_neighbor_map.data_ptr<int>();
  int *kpos_ptr = neighbor_address.data_ptr<int>();
  int *qkpos_ptr = q_neighbor_address.data_ptr<int>();
  int *in_map_ptr;
  int *out_map_ptr;
  if (transpose){
    in_map_ptr = neighbor_map.data_ptr<int>() + sum_nnz;
    out_map_ptr = neighbor_map.data_ptr<int>();
  }
  else{
    in_map_ptr = neighbor_map.data_ptr<int>();
    out_map_ptr = neighbor_map.data_ptr<int>() + sum_nnz;
  }

  // memory allocation
  at::Tensor out_feat = torch::zeros({output_size, out_channel}, 
            at::device(in_feat.device()).dtype(in_feat.scalar_type()));
  // at::Tensor kpos = torch::zeros({k_vol + 1}, 
  //           at::device(in_feat.device()).dtype(at::ScalarType::Int));
  // at::Tensor qkpos = torch::zeros({k_vol + 1}, 
  //           at::device(in_feat.device()).dtype(at::ScalarType::Int));
  // int *kpos_ptr = kpos.data_ptr<int>();
  // int *qkpos_ptr = qkpos.data_ptr<int>();

  // should be modified in the future
  int mid_kernel = (k_vol % 2 == 1) ? k_vol / 2 : 0;

  bool data_type_half = in_feat.scalar_type() == at::ScalarType::Half;
  // bool precompute_mid = (input_size == output_size && k_vol % 2 == 1);
  bool precompute_mid = false;

  // exclusive_scan_for_kernel_quantified<<<1, k_vol, 0, 0>>>(
  //       k_vol + 1, knnz_ptr, 128, kpos_ptr, qkpos_ptr
  // );

  // int qsum_nnz = qkpos[k_vol].item<int>();
  // printf("%d", qsum_nnz);

  if (data_type_half && allow_fp16){
    if (in_channel % 4 == 0 && out_channel % 4 == 0){    
      if (in_channel <= 16 || out_channel <= 16){
        fetch_on_demand_gemm_fp16_4_once<16, 4, 8>
            <<<dim3(DIV_UP(out_channel, 16), DIV_UP(qsum_nnz, 64), 1),
               dim3(4, 16, 1)>>>(
                kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
                reinterpret_cast<sycl::half *>(in_feat.data_ptr<at::Half>()),
                reinterpret_cast<sycl::half *>(kernel.data_ptr<at::Half>()),
                reinterpret_cast<sycl::half *>(out_feat.data_ptr<at::Half>()),
                in_map_ptr, out_map_ptr);
      }
      else{
        if (allow_tf32){
          fetch_on_demand_gemm_fp16_tc4_async<32, 4, 8, 16, 16, 16, 4, 2, 2>
              <<<dim3(DIV_UP(out_channel, 32), DIV_UP(qsum_nnz, 128), 1),
                 dim3(8, 32, 1)>>>(
                  kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
                  reinterpret_cast<sycl::half *>(in_feat.data_ptr<at::Half>()),
                  reinterpret_cast<sycl::half *>(kernel.data_ptr<at::Half>()),
                  reinterpret_cast<sycl::half *>(out_feat.data_ptr<at::Half>()),
                  in_map_ptr, out_map_ptr);
        }
        else{
          fetch_on_demand_gemm_fp16_tc4<32, 4, 8, 16, 16, 16, 4, 2, 2>
              <<<dim3(DIV_UP(out_channel, 32), DIV_UP(qsum_nnz, 128), 1),
                 dim3(8, 32, 1)>>>(
                  kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
                  reinterpret_cast<sycl::half *>(in_feat.data_ptr<at::Half>()),
                  reinterpret_cast<sycl::half *>(kernel.data_ptr<at::Half>()),
                  reinterpret_cast<sycl::half *>(out_feat.data_ptr<at::Half>()),
                  in_map_ptr, out_map_ptr);
        }
      }
    }
    else if (in_channel % 2 == 0 && out_channel % 2 == 0){
        fetch_on_demand_gemm_fp16_2<16, 8, 8>
            <<<dim3(DIV_UP(out_channel, 16), DIV_UP(qsum_nnz, 128), 1),
               dim3(8, 16, 1)>>>(
                kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
                reinterpret_cast<sycl::half *>(in_feat.data_ptr<at::Half>()),
                reinterpret_cast<sycl::half *>(kernel.data_ptr<at::Half>()),
                reinterpret_cast<sycl::half *>(out_feat.data_ptr<at::Half>()),
                in_map_ptr, out_map_ptr);
    }
    else{
        fetch_on_demand_gemm_fp16_1<16, 4, 8>
            <<<dim3(DIV_UP(out_channel, 16), DIV_UP(qsum_nnz, 64), 1),
               dim3(16, 16, 1)>>>(
                kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
                reinterpret_cast<sycl::half *>(in_feat.data_ptr<at::Half>()),
                reinterpret_cast<sycl::half *>(kernel.data_ptr<at::Half>()),
                reinterpret_cast<sycl::half *>(out_feat.data_ptr<at::Half>()),
                in_map_ptr, out_map_ptr);
    }  
  }
  else{
    if(in_channel % 4 == 0 && out_channel % 4 ==0){
      if (in_channel <= 16 && out_channel <= 16){
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, DIV_UP(qsum_nnz, 64),
                                             DIV_UP(out_channel, 16)) *
                                  sycl::range<3>(1, 16, 4),
                              sycl::range<3>(1, 16, 4)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
              fetch_on_demand_gemm_fp32_once<16, 4, 8>(
                  kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
                  in_feat.data_ptr<float>(), kernel.data_ptr<float>(),
                  out_feat.data_ptr<float>(), in_map_ptr, out_map_ptr);
            });
      }
      else{
        if (allow_tf32){
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, DIV_UP(qsum_nnz, 128),
                                               DIV_UP(out_channel, 32)) *
                                    sycl::range<3>(1, 32, 8),
                                sycl::range<3>(1, 32, 8)),
              exp_props, [=](sycl::nd_item<3> item_ct1) {
                fetch_on_demand_gemm_tf32<32, 4, 8, 16, 8, 16, 4, 2, 2>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
                    in_feat.data_ptr<float>(), kernel.data_ptr<float>(),
                    out_feat.data_ptr<float>(), in_map_ptr, out_map_ptr);
              });
        }
        else{
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, DIV_UP(qsum_nnz, 128),
                                               DIV_UP(out_channel, 32)) *
                                    sycl::range<3>(1, 32, 8),
                                sycl::range<3>(1, 32, 8)),
              exp_props, [=](sycl::nd_item<3> item_ct1) {
                fetch_on_demand_gemm_fp32<32, 4, 8>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
                    in_feat.data_ptr<float>(), kernel.data_ptr<float>(),
                    out_feat.data_ptr<float>(), in_map_ptr, out_map_ptr);
              });
        }
      }
    }
    else if (in_channel % 2 == 0 && out_channel % 2 == 0){
      auto exp_props = sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::use_root_sync};
      q_ct1.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, DIV_UP(qsum_nnz, 128),
                                           DIV_UP(out_channel, 16)) *
                                sycl::range<3>(1, 16, 8),
                            sycl::range<3>(1, 16, 8)),
          exp_props, [=](sycl::nd_item<3> item_ct1) {
            fetch_on_demand_gemm_fp32_2<16, 8, 8>(
                kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
                in_feat.data_ptr<float>(), kernel.data_ptr<float>(),
                out_feat.data_ptr<float>(), in_map_ptr, out_map_ptr);
          });
    }
    else{
      auto exp_props = sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::use_root_sync};
      q_ct1.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, DIV_UP(qsum_nnz, 64), DIV_UP(out_channel, 16)) *
                  sycl::range<3>(1, 16, 16),
              sycl::range<3>(1, 16, 16)),
          exp_props, [=](sycl::nd_item<3> item_ct1) {
            fetch_on_demand_gemm_fp32_1<16, 4, 8>(
                kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
                in_feat.data_ptr<float>(), kernel.data_ptr<float>(),
                out_feat.data_ptr<float>(), in_map_ptr, out_map_ptr);
          });
    }
  }

  // precomputation only for odd channel size
  // bool precompute_mid = (input_size == output_size && k_vol % 2 == 1);
  if (precompute_mid){
    at::addmm_out(out_feat, out_feat, in_feat, kernel[mid_kernel]);
  }

  return out_feat;
}

at::Tensor conv_forward_fetch_on_demand_no_fusion_cuda(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, const int sum_nnz, const int output_size,
    const bool transpose, const bool allow_tf32, const bool allow_fp16) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();

  // int sum_nnz = (int)torch::sum(neighbor_offset).item<int>();
  int input_size = in_feat.size(0);
  int in_channel = in_feat.size(1);
  int out_channel = kernel.size(2);
  int k_vol = kernel.size(0);
  int *knnz_ptr = neighbor_offset.data_ptr<int>();
  // int *in_map_ptr = in_neighbor_map.data_ptr<int>();
  // int *out_map_ptr = out_neighbor_map.data_ptr<int>();
  // int *kpos_ptr = neighbor_address.data_ptr<int>();
  // int *qkpos_ptr = q_neighbor_address.data_ptr<int>();
  int *in_map_ptr;
  int *out_map_ptr;
  if (transpose){
    in_map_ptr = neighbor_map.data_ptr<int>() + sum_nnz;
    out_map_ptr = neighbor_map.data_ptr<int>();
  }
  else{
    in_map_ptr = neighbor_map.data_ptr<int>();
    out_map_ptr = neighbor_map.data_ptr<int>() + sum_nnz;
  }

  // memory allocation
  at::Tensor out_feat = torch::zeros({output_size, out_channel}, 
            at::device(in_feat.device()).dtype(in_feat.scalar_type()));
  // at::Tensor kpos = torch::zeros({k_vol + 1}, 
  //           at::device(in_feat.device()).dtype(at::ScalarType::Int));
  // at::Tensor qkpos = torch::zeros({k_vol + 1}, 
  //           at::device(in_feat.device()).dtype(at::ScalarType::Int));
  // int *kpos_ptr = kpos.data_ptr<int>();
  // int *qkpos_ptr = qkpos.data_ptr<int>();

  // should be modified in the future
  int mid_kernel = (k_vol % 2 == 1) ? k_vol / 2 : 0;

  bool data_type_half = in_feat.scalar_type() == at::ScalarType::Half;
  // bool precompute_mid = (input_size == output_size && k_vol % 2 == 1);
  bool precompute_mid = false;

  /********************************************************************/
  // loop over all kernel offsets
  int cur_idx = 0;
  // int stream_id = 0;
  for (int k = 0; k < k_vol; k++){
    int cur_nnz = knnz_ptr[k];
    
    if (cur_nnz == 0){continue;}

    // size_t gridnum_x = DIV_UP(out_channel, 32);
    // size_t gridnum_y = DIV_UP(cur_nnz, 32);

    if (data_type_half && allow_fp16){
      if (in_channel % 4 == 0 && out_channel % 4 == 0){
        fetch_on_demand_gemm_no_fusion_fp16<32, 4, 8, 16, 16, 16, 4, 2, 2>
            <<<dim3(DIV_UP(out_channel, 32), DIV_UP(cur_nnz, 32), 1),
               dim3(8, 32, 1)>>>(
                cur_nnz, in_channel, out_channel,
                reinterpret_cast<sycl::half *>(in_feat.data_ptr<at::Half>()),
                reinterpret_cast<sycl::half *>(kernel.data_ptr<at::Half>() +
                                               k * in_channel * out_channel),
                reinterpret_cast<sycl::half *>(out_feat.data_ptr<at::Half>()),
                &in_map_ptr[cur_idx], &out_map_ptr[cur_idx]);
      }
      else{
        fetch_on_demand_gemm_no_fusion_fp16_1<16, 4, 8>
            <<<dim3(DIV_UP(out_channel, 16), DIV_UP(cur_nnz, 16), 1),
               dim3(16, 16, 1)>>>(
                cur_nnz, in_channel, out_channel,
                reinterpret_cast<sycl::half *>(in_feat.data_ptr<at::Half>()),
                reinterpret_cast<sycl::half *>(kernel.data_ptr<at::Half>() +
                                               k * in_channel * out_channel),
                reinterpret_cast<sycl::half *>(out_feat.data_ptr<at::Half>()),
                &in_map_ptr[cur_idx], &out_map_ptr[cur_idx]);
      }
    }
    else{
      if (in_channel % 4 == 0 && out_channel % 4 == 0){
        if (allow_tf32){
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            auto kernel_data_ptr_float_k_in_channel_out_channel_ct4 =
                (kernel.data_ptr<float>() + k * in_channel * out_channel);
            auto in_map_ptr_cur_idx_ct6 = &in_map_ptr[cur_idx];
            auto out_map_ptr_cur_idx_ct7 = &out_map_ptr[cur_idx];

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, DIV_UP(cur_nnz, 32),
                                                 DIV_UP(out_channel, 32)) *
                                      sycl::range<3>(1, 32, 8),
                                  sycl::range<3>(1, 32, 8)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  fetch_on_demand_gemm_no_fusion_tf32<32, 4, 8, 16, 8, 16, 4, 2,
                                                      2>(
                      cur_nnz, in_channel, out_channel,
                      in_feat.data_ptr<float>(),
                      kernel_data_ptr_float_k_in_channel_out_channel_ct4,
                      out_feat.data_ptr<float>(), in_map_ptr_cur_idx_ct6,
                      out_map_ptr_cur_idx_ct7);
                });
          });
        }
        else{
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            auto kernel_data_ptr_float_k_in_channel_out_channel_ct4 =
                (kernel.data_ptr<float>() + k * in_channel * out_channel);
            auto in_map_ptr_cur_idx_ct6 = &in_map_ptr[cur_idx];
            auto out_map_ptr_cur_idx_ct7 = &out_map_ptr[cur_idx];

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, DIV_UP(cur_nnz, 32),
                                                 DIV_UP(out_channel, 32)) *
                                      sycl::range<3>(1, 32, 8),
                                  sycl::range<3>(1, 32, 8)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  fetch_on_demand_gemm_no_fusion_fp32<32, 4, 8>(
                      cur_nnz, in_channel, out_channel,
                      in_feat.data_ptr<float>(),
                      kernel_data_ptr_float_k_in_channel_out_channel_ct4,
                      out_feat.data_ptr<float>(), in_map_ptr_cur_idx_ct6,
                      out_map_ptr_cur_idx_ct7);
                });
          });
        }
      }
      else{
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        q_ct1.submit([&](sycl::handler &cgh) {
          auto kernel_data_ptr_float_k_in_channel_out_channel_ct4 =
              (kernel.data_ptr<float>() + k * in_channel * out_channel);
          auto in_map_ptr_cur_idx_ct6 = &in_map_ptr[cur_idx];
          auto out_map_ptr_cur_idx_ct7 = &out_map_ptr[cur_idx];

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, DIV_UP(cur_nnz, 16),
                                               DIV_UP(out_channel, 16)) *
                                    sycl::range<3>(1, 16, 16),
                                sycl::range<3>(1, 16, 16)),
              exp_props, [=](sycl::nd_item<3> item_ct1) {
                fetch_on_demand_gemm_no_fusion_fp32_1<16, 4, 8>(
                    cur_nnz, in_channel, out_channel, in_feat.data_ptr<float>(),
                    kernel_data_ptr_float_k_in_channel_out_channel_ct4,
                    out_feat.data_ptr<float>(), in_map_ptr_cur_idx_ct6,
                    out_map_ptr_cur_idx_ct7);
              });
        });
      }
    }

    cur_idx += cur_nnz;
  }
  // precomputation only for odd channel size
  // bool precompute_mid = (input_size == output_size && k_vol % 2 == 1);
  if (precompute_mid){
    at::addmm_out(out_feat, out_feat, in_feat, kernel[mid_kernel]);
  }

  return out_feat;
}

