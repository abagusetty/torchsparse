#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/extension.h>
#include "convolution_backward_wgrad_implicit_gemm_sorted_cuda.h"
#include "../utils/memory.dp.hpp"
#include <stdexcept>
#include <sycl/ext/intel/math.hpp>

// Pack two half values.
static inline unsigned __pack_half2(const sycl::half x, const sycl::half y)
{
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}


// conv_backward_cuda_m16n16k64_m16n16k64_m16n16k16_f16f16f32
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
/*
DPCT1110:0: The total declared local variable size in device function
conv_backward_cuda_setting1_mode1_f16f16f32 exceeds 128 bytes and may cause high
register pressure. Consult with your hardware vendor to find the total register
size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void conv_backward_cuda_setting1_mode1_f16f16f32(
    int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters,
    int split_mask_len, int reduced_mask_len, int reorder_loc_len,
    sycl::half *__restrict__ A, sycl::half *__restrict__ B,
    int *__restrict__ reduced_mask, int *__restrict__ out_in_map,
    int *__restrict__ reorder_loc, sycl::half *__restrict__ C)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int K_tile = 16;
  int K_tile_padded = K_tile * ((K_original + K_tile - 1) / K_tile);

  float C_warp[8];
  auto &A_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[2560]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  auto &B_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[2560]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  sycl::half A_shared_warp[8];
  sycl::half B_shared_warp[8];
  for (int i = 0; i < 8; ++i)
  {
    C_warp[0 + i] = 0.0;
  }

  // hoisting shared pointer offsets
  int j_factors1 = (N + 15) / 16 / 1;
  int blockIdx_x = 0;
  int blockIdx_y = item_ct1.get_group(2) %
                   ((K_original + 15) / 16 * kernel_volume * j_factors1);
  int blockIdx_z = item_ct1.get_group(2) /
                   ((K_original + 15) / 16 * kernel_volume * j_factors1);
  sycl::half *cur_C = C + blockIdx_z * kernel_volume * K_original * N;
  int *out_in_map_ptr =
      out_in_map +
      (item_ct1.get_local_id(1) * 16 + item_ct1.get_local_id(2) / 2) *
          kernel_volume +
      ((item_ct1.get_local_id(1) * 256) % 16) / K_tile_padded +
      ((item_ct1.get_local_id(2) * 8) % 16) / K_tile_padded +
      (blockIdx_y / j_factors1 * 16) / K_tile_padded;
  sycl::half *A_ptr = A +
                      ((item_ct1.get_local_id(1) * 256 % 16) % K_tile_padded) +
                      ((item_ct1.get_local_id(2) * 8 % 16) % K_tile_padded) +
                      ((blockIdx_y / j_factors1 * 16) % K_tile_padded);
  sycl::half *B_ptr =
      B + (blockIdx_y % j_factors1) * 16 + (item_ct1.get_local_id(2) * 8) % 16;
  int reorder_offset =
      item_ct1.get_local_id(1) * 256 / 16 + item_ct1.get_local_id(2) * 8 / 16;
  int K_iters = ((M_fwd + 63) / 64 + split_k_iters - 1) / split_k_iters;
  int kernel_offset = (blockIdx_y / j_factors1) / ((K_original + K_tile - 1) / K_tile);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_ptr = reorder_loc + split_mask_iter * reorder_loc_len;
  int* reduced_mask_ptr = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;
  int cur_C_ic_start = (blockIdx_y / j_factors1 * 16) % K_tile_padded +
                       (item_ct1.get_local_id(2) / 4);
  int cur_C_oc_start = (blockIdx_y % j_factors1) * 16 +
                       item_ct1.get_local_id(1) / 1 * 16 +
                       (item_ct1.get_local_id(2) % 4) * 2;
  sycl::half *C_ptr = cur_C +
                      (kernel_offset * K_original + cur_C_ic_start) * N +
                      cur_C_oc_start;

  int A_pred_guard = 0;
  int B_pred_guard = 0;
  if constexpr (K_ld_check)
  {
    int A_ld_start = ((item_ct1.get_local_id(1) * 256 % 16) % K_tile_padded) +
                     ((item_ct1.get_local_id(2) * 8 % 16) % K_tile_padded) +
                     ((blockIdx_y / j_factors1 * 16) % K_tile_padded);
    int A_ld_amount = sycl::min(A_ld_start + 8, K_original) - A_ld_start;
    int A_ld_bound = A_ld_amount / (K_ld_factor / 2);

    for (int i = 0; i < A_ld_bound; i++)
      A_pred_guard |= (1 << i);
  }
  else
    A_pred_guard = 1;
  if constexpr (N_ld_check)
  {
    int B_ld_start =
        (blockIdx_y % j_factors1) * 16 + (item_ct1.get_local_id(2) * 8) % 16;
    int B_ld_amount = sycl::min(B_ld_start + 8, N) - B_ld_start;
    int B_ld_bound = B_ld_amount / (N_ld_factor / 2);

    for (int i = 0; i < B_ld_bound; i++)
      B_pred_guard |= (1 << i);
  }
  else
    B_pred_guard = 1;


  for (int _i2_0_0 = 0; _i2_0_0 < K_iters - 1; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;
    
    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    sycl::half *A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    /*
    DPCT1118:1: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume 
        + (ax0_ax1_fused_0 * 256 % 16) / K_tile_padded
      ];

      if (input_idx != -1)
      {
        sycl::uint4 A_loaded = sycl::uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded), A_pred_guard);
        *(sycl::uint4 *)(A_shared +
                         (((ax0_ax1_fused_0 * 640) +
                           ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                          ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            A_loaded;
      }
      else
      {
        *(sycl::uint4 *)(A_shared +
                         (((ax0_ax1_fused_0 * 640) +
                           ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                          ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            sycl::uint4(
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
    {
  
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 16;
      int v0 = reorder_loc_ptr[reorder_offset_inner];
      sycl::uint4 B_loaded = sycl::uint4(0, 0, 0, 0);
      global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
      *(sycl::uint4 *)(B_shared +
                       (((ax0_ax1_fused_0_1 * 640) +
                         ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                        ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
          B_loaded;
    }

    /*
    DPCT1118:2: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {

      {
        unsigned int addr;
        /*
        DPCT1053:3: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
            "addr; }"
            : "=r"(addr)
            : "l"((void *)((&(A_shared[(i2_0_1 * 640)])) +
                           (((((int)item_ct1.get_local_id(2)) & 15) * 40) +
                            ((((int)item_ct1.get_local_id(2)) >> 4) * 8)))));
#if DPCT_COMPATIBILITY_TEMP >= 750
        /*
        DPCT1053:4: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
                             "{%0, %1, %2, %3}, [%4];"
                             : "=r"(((unsigned *)(A_shared_warp + 0))[0]),
                               "=r"(((unsigned *)(A_shared_warp + 0))[2]),
                               "=r"(((unsigned *)(A_shared_warp + 0))[1]),
                               "=r"(((unsigned *)(A_shared_warp + 0))[3])
                             : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }

      {
        unsigned int addr;
        /*
        DPCT1053:5: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
            "addr; }"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[(i2_0_1 * 640)])) +
                           (((((int)item_ct1.get_local_id(2)) & 15) * 40) +
                            ((((int)item_ct1.get_local_id(2)) >> 4) * 8)))));
#if DPCT_COMPATIBILITY_TEMP >= 750
        /*
        DPCT1053:6: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
                             "{%0, %1, %2, %3}, [%4];"
                             : "=r"(((unsigned *)(B_shared_warp + 0))[0]),
                               "=r"(((unsigned *)(B_shared_warp + 0))[1]),
                               "=r"(((unsigned *)(B_shared_warp + 0))[2]),
                               "=r"(((unsigned *)(B_shared_warp + 0))[3])
                             : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }
#if DPCT_COMPATIBILITY_TEMP >= 800
      {
        /*
        DPCT1053:7: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 0))[0]),
              "=f"(((float *)(C_warp + 0))[1]),
              "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]),
              "r"(((unsigned *)(A_shared_warp + 0))[1]),
              "r"(((unsigned *)(A_shared_warp + 0))[2]),
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + 0))[0]),
              "r"(((unsigned *)(B_shared_warp + 0))[1]),
              "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]),
              "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        /*
        DPCT1053:8: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 4))[0]),
              "=f"(((float *)(C_warp + 4))[1]),
              "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]),
              "r"(((unsigned *)(A_shared_warp + 0))[1]),
              "r"(((unsigned *)(A_shared_warp + 0))[2]),
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + 4))[0]),
              "r"(((unsigned *)(B_shared_warp + 4))[1]),
              "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]),
              "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#elif DPCT_COMPATIBILITY_TEMP >= 750
      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 2))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 6))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
    }
  }
  
  for (int _i2_0_0 = K_iters - 1; _i2_0_0 < K_iters; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * (K_iters - 1);
    if (i2_0_0 >= (M_fwd + 63) / 64)
      continue;
  
    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    sycl::half *A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    /*
    DPCT1118:9: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume 
        + (ax0_ax1_fused_0 * 256 % 16) / K_tile_padded
      ];

      if (input_idx != -1)
      {
        sycl::uint4 A_loaded = sycl::uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded), A_pred_guard);
        *(sycl::uint4 *)(A_shared +
                         (((ax0_ax1_fused_0 * 640) +
                           ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                          ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            A_loaded;
      }
      else
      {
        *(sycl::uint4 *)(A_shared +
                         (((ax0_ax1_fused_0 * 640) +
                           ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                          ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            sycl::uint4(
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
    {
  
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 16;
      if (reorder_offset_inner < M_fwd){
        int v0 = reorder_loc_ptr[reorder_offset_inner];
        sycl::uint4 B_loaded = sycl::uint4(0, 0, 0, 0);
        global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
        *(sycl::uint4 *)(B_shared +
                         (((ax0_ax1_fused_0_1 * 640) +
                           ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                          ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            B_loaded;
        }
        else 
        {
          *(sycl::uint4 *)(B_shared +
                           (((ax0_ax1_fused_0_1 * 640) +
                             ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                            ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
              sycl::uint4(0, 0, 0, 0);
        }
      }

    /*
    DPCT1118:10: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {

      {
        unsigned int addr;
        /*
        DPCT1053:11: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
            "addr; }"
            : "=r"(addr)
            : "l"((void *)((&(A_shared[(i2_0_1 * 640)])) +
                           (((((int)item_ct1.get_local_id(2)) & 15) * 40) +
                            ((((int)item_ct1.get_local_id(2)) >> 4) * 8)))));
#if DPCT_COMPATIBILITY_TEMP >= 750
        /*
        DPCT1053:12: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
                             "{%0, %1, %2, %3}, [%4];"
                             : "=r"(((unsigned *)(A_shared_warp + 0))[0]),
                               "=r"(((unsigned *)(A_shared_warp + 0))[2]),
                               "=r"(((unsigned *)(A_shared_warp + 0))[1]),
                               "=r"(((unsigned *)(A_shared_warp + 0))[3])
                             : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }

      {
        unsigned int addr;
        /*
        DPCT1053:13: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
            "addr; }"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[(i2_0_1 * 640)])) +
                           (((((int)item_ct1.get_local_id(2)) & 15) * 40) +
                            ((((int)item_ct1.get_local_id(2)) >> 4) * 8)))));
#if DPCT_COMPATIBILITY_TEMP >= 750
        /*
        DPCT1053:14: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
                             "{%0, %1, %2, %3}, [%4];"
                             : "=r"(((unsigned *)(B_shared_warp + 0))[0]),
                               "=r"(((unsigned *)(B_shared_warp + 0))[1]),
                               "=r"(((unsigned *)(B_shared_warp + 0))[2]),
                               "=r"(((unsigned *)(B_shared_warp + 0))[3])
                             : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }
#if DPCT_COMPATIBILITY_TEMP >= 800
      {
        /*
        DPCT1053:15: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 0))[0]),
              "=f"(((float *)(C_warp + 0))[1]),
              "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]),
              "r"(((unsigned *)(A_shared_warp + 0))[1]),
              "r"(((unsigned *)(A_shared_warp + 0))[2]),
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + 0))[0]),
              "r"(((unsigned *)(B_shared_warp + 0))[1]),
              "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]),
              "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        /*
        DPCT1053:16: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 4))[0]),
              "=f"(((float *)(C_warp + 4))[1]),
              "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]),
              "r"(((unsigned *)(A_shared_warp + 0))[1]),
              "r"(((unsigned *)(A_shared_warp + 0))[2]),
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + 4))[0]),
              "r"(((unsigned *)(B_shared_warp + 4))[1]),
              "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]),
              "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#elif DPCT_COMPATIBILITY_TEMP >= 750
      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 2))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 6))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
    }
  }
  
  for (int local_id = 0; local_id < 8; ++local_id)
  {
    if constexpr (K_ld_check || N_ld_check)
    {
      if (cur_C_ic_start + ((local_id / 2) % 2) * 8 < K_original && cur_C_oc_start + (local_id % 2) + (local_id / 4) * 8 < N)
        C_ptr[+(((local_id / 2) % 2) * 8) * N + (local_id % 2) +
              (local_id / 4) * 8] =
            sycl::ext::intel::math::float2half_rn(C_warp[0 + local_id]);
    }
    else
    {
      C_ptr[+(((local_id / 2) % 2) * 8) * N + (local_id % 2) +
            (local_id / 4) * 8] =
          sycl::ext::intel::math::float2half_rn(C_warp[0 + local_id]);
    }
  }
}


// conv_backward_cuda_m32n64k64_m32n32k64_m16n16k16_f16f16f32
/*
DPCT1110:17: The total declared local variable size in device function
conv_backward_cuda_setting2_mode1_f16f16f32 exceeds 128 bytes and may cause high
register pressure. Consult with your hardware vendor to find the total register
size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void conv_backward_cuda_setting2_mode1_f16f16f32(
    int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters,
    int split_mask_len, int reduced_mask_len, int reorder_loc_len,
    sycl::half *__restrict__ A, sycl::half *__restrict__ B,
    int *__restrict__ reduced_mask, int *__restrict__ out_in_map,
    int *__restrict__ reorder_loc, sycl::half *__restrict__ C)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  float C_warp[32];
  auto &A_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[2560]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  auto &B_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[4608]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  sycl::half A_shared_warp[16];
  sycl::half B_shared_warp[16];
  for (int i0_0_3_init = 0; i0_0_3_init < 2; ++i0_0_3_init)
  {
    for (int i1_0_4_init = 0; i1_0_4_init < 2; ++i1_0_4_init)
    {
      for (int i = 0; i < 8; ++i)
      {
        C_warp[((i0_0_3_init * 16) + (i1_0_4_init * 8)) + i] = 0.0;
      };
    }
  }

  // hoisting shared pointer offsets
  int j_factors1 = N / 16 / 4;
  int blockIdx_x = 0;
  int blockIdx_y = item_ct1.get_group(2) %
                   ((K_original * kernel_volume + 31) / 32 * j_factors1);
  int blockIdx_z = item_ct1.get_group(2) /
                   ((K_original * kernel_volume + 31) / 32 * j_factors1);
  sycl::half *cur_C = C + blockIdx_z * kernel_volume * N * K_original;
  int *out_in_map_ptr =
      out_in_map +
      (item_ct1.get_local_id(1) * 8 + item_ct1.get_local_id(2) / 4) *
          kernel_volume +
      ((item_ct1.get_local_id(1) * 256) % 32) / K_original +
      ((item_ct1.get_local_id(2) * 8) % 32) / K_original +
      (blockIdx_y / j_factors1 * 32) / K_original;
  sycl::half *A_ptr = A + ((item_ct1.get_local_id(1) * 256 % 32) % K_original) +
                      ((item_ct1.get_local_id(2) * 8 % 32) % K_original) +
                      ((blockIdx_y / j_factors1 * 32) % K_original);
  sycl::half *B_ptr =
      B + (blockIdx_y % j_factors1) * 64 + (item_ct1.get_local_id(2) * 8) % 64;
  int reorder_offset =
      item_ct1.get_local_id(1) * 256 / 64 + item_ct1.get_local_id(2) * 8 / 64;
  sycl::half *C_ptr =
      cur_C + blockIdx_x / 1 * 108 * N / 16 * 256 +
      blockIdx_y / j_factors1 * 2 * N / 16 * 256 +
      (item_ct1.get_local_id(1) % 1) * 2 * N / 16 * 256 +
      (blockIdx_x % 1) * j_factors1 * 64 + (blockIdx_y % j_factors1) * 64 +
      item_ct1.get_local_id(1) / 1 * 32 + (item_ct1.get_local_id(2) % 4) * 2 +
      (item_ct1.get_local_id(2) / 4) * N;
  int K_iters = ((M_fwd + 63) / 64 + split_k_iters - 1) / split_k_iters;
  int kernel_offset = (blockIdx_y / j_factors1) / (K_original / 32);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_ptr = reorder_loc + split_mask_iter * reorder_loc_len;
  int* reduced_mask_ptr = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;

  for (int _i2_0_0 = 0; _i2_0_0 < K_iters - 1; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;
    
    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    sycl::half *A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    /*
    DPCT1118:18: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume 
        + (ax0_ax1_fused_0 * 512 % 32) / K_original
      ];

      if (input_idx != -1)
      {
        *(sycl::uint4 *)(A_shared +
                         ((((ax0_ax1_fused_0 * 640) +
                            (((int)item_ct1.get_local_id(1)) * 320)) +
                           ((((int)item_ct1.get_local_id(2)) >> 2) * 40)) +
                          ((((int)item_ct1.get_local_id(2)) & 3) * 8))) =
            *(sycl::uint4 *)(A_ptr_local + input_idx * K_original +
                             ((ax0_ax1_fused_0 * 512 % 32) % K_original));
      }
      else
      {
        *(sycl::uint4 *)(A_shared +
                         ((((ax0_ax1_fused_0 * 640) +
                            (((int)item_ct1.get_local_id(1)) * 320)) +
                           ((((int)item_ct1.get_local_id(2)) >> 2) * 40)) +
                          ((((int)item_ct1.get_local_id(2)) & 3) * 8))) =
            sycl::uint4(
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
  
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 8;
      int v0 = reorder_loc_ptr[reorder_offset_inner];
      *(sycl::uint4 *)(B_shared +
                       ((((ax0_ax1_fused_0_1 * 576) +
                          (((int)item_ct1.get_local_id(1)) * 288)) +
                         ((((int)item_ct1.get_local_id(2)) >> 3) * 72)) +
                        ((((int)item_ct1.get_local_id(2)) & 7) * 8))) =
          *(sycl::uint4 *)(B_ptr + v0 * N);
    }

    /*
    DPCT1118:19: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {

        {
          unsigned int addr;
          /*
          DPCT1053:20: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
              "addr; }"
              : "=r"(addr)
              : "l"((void *)((&(A_shared[((i2_0_1 * 640) + (ax1_0 * 16))])) +
                             (((((int)item_ct1.get_local_id(2)) & 15) * 40) +
                              ((((int)item_ct1.get_local_id(2)) >> 4) * 8)))));
#if DPCT_COMPATIBILITY_TEMP >= 750
          /*
          DPCT1053:21: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[0]),
                "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[2]),
                "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[1]),
                "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
      {

        {
          unsigned int addr;
          /*
          DPCT1053:22: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
              "addr; }"
              : "=r"(addr)
              : "l"((void *)((&(B_shared[(
                                 ((i2_0_1 * 1152) +
                                  (((int)item_ct1.get_local_id(1)) * 32)) +
                                 (ax1_0_1 * 16))])) +
                             (((((int)item_ct1.get_local_id(2)) & 15) * 72) +
                              ((((int)item_ct1.get_local_id(2)) >> 4) * 8)))));
#if DPCT_COMPATIBILITY_TEMP >= 750
          /*
          DPCT1053:23: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[0]),
                "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[1]),
                "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[2]),
                "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int i0_0_3 = 0; i0_0_3 < 2; ++i0_0_3)
      {
        for (int i1_0_4 = 0; i1_0_4 < 2; ++i1_0_4)
        {
#if DPCT_COMPATIBILITY_TEMP >= 800
          {
            /*
            DPCT1053:24: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]),
                  "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]),
                  "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            /*
            DPCT1053:25: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#elif DPCT_COMPATIBILITY_TEMP >= 750
          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
    }
  }
  
  for (int _i2_0_0 = K_iters - 1; _i2_0_0 < K_iters; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * (K_iters - 1);
    if (i2_0_0 >= (M_fwd + 63) / 64)
      continue;
  
    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    sycl::half *A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    /*
    DPCT1118:26: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume 
        + (ax0_ax1_fused_0 * 512 % 32) / K_original
      ];

      if (input_idx != -1)
      {
        *(sycl::uint4 *)(A_shared +
                         ((((ax0_ax1_fused_0 * 640) +
                            (((int)item_ct1.get_local_id(1)) * 320)) +
                           ((((int)item_ct1.get_local_id(2)) >> 2) * 40)) +
                          ((((int)item_ct1.get_local_id(2)) & 3) * 8))) =
            *(sycl::uint4 *)(A_ptr_local + input_idx * K_original +
                             ((ax0_ax1_fused_0 * 512 % 32) % K_original));
      }
      else
      {
        *(sycl::uint4 *)(A_shared +
                         ((((ax0_ax1_fused_0 * 640) +
                            (((int)item_ct1.get_local_id(1)) * 320)) +
                           ((((int)item_ct1.get_local_id(2)) >> 2) * 40)) +
                          ((((int)item_ct1.get_local_id(2)) & 3) * 8))) =
            sycl::uint4(
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)),
                __pack_half2(
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f),
                    sycl::ext::intel::math::float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
  
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 8;
      if (reorder_offset_inner < M_fwd){
        int v0 = reorder_loc_ptr[reorder_offset_inner];
        *(sycl::uint4 *)(B_shared +
                         ((((ax0_ax1_fused_0_1 * 576) +
                            (((int)item_ct1.get_local_id(1)) * 288)) +
                           ((((int)item_ct1.get_local_id(2)) >> 3) * 72)) +
                          ((((int)item_ct1.get_local_id(2)) & 7) * 8))) =
            *(sycl::uint4 *)(B_ptr + v0 * N);
        }
        else 
        {
          *(sycl::uint4 *)(B_shared +
                           ((((ax0_ax1_fused_0_1 * 576) +
                              (((int)item_ct1.get_local_id(1)) * 288)) +
                             ((((int)item_ct1.get_local_id(2)) >> 3) * 72)) +
                            ((((int)item_ct1.get_local_id(2)) & 7) * 8))) =
              sycl::uint4(0, 0, 0, 0);
        }
      }

    /*
    DPCT1118:27: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {

        {
          unsigned int addr;
          /*
          DPCT1053:28: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
              "addr; }"
              : "=r"(addr)
              : "l"((void *)((&(A_shared[((i2_0_1 * 640) + (ax1_0 * 16))])) +
                             (((((int)item_ct1.get_local_id(2)) & 15) * 40) +
                              ((((int)item_ct1.get_local_id(2)) >> 4) * 8)))));
#if DPCT_COMPATIBILITY_TEMP >= 750
          /*
          DPCT1053:29: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[0]),
                "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[2]),
                "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[1]),
                "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
      {

        {
          unsigned int addr;
          /*
          DPCT1053:30: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
              "addr; }"
              : "=r"(addr)
              : "l"((void *)((&(B_shared[(
                                 ((i2_0_1 * 1152) +
                                  (((int)item_ct1.get_local_id(1)) * 32)) +
                                 (ax1_0_1 * 16))])) +
                             (((((int)item_ct1.get_local_id(2)) & 15) * 72) +
                              ((((int)item_ct1.get_local_id(2)) >> 4) * 8)))));
#if DPCT_COMPATIBILITY_TEMP >= 750
          /*
          DPCT1053:31: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[0]),
                "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[1]),
                "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[2]),
                "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int i0_0_3 = 0; i0_0_3 < 2; ++i0_0_3)
      {
        for (int i1_0_4 = 0; i1_0_4 < 2; ++i1_0_4)
        {
#if DPCT_COMPATIBILITY_TEMP >= 800
          {
            /*
            DPCT1053:32: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]),
                  "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]),
                  "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            /*
            DPCT1053:33: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#elif DPCT_COMPATIBILITY_TEMP >= 750
          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
    }
  }
  
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0)
  {

    sycl::half *C_ptr_local = C_ptr + ax0_0 * N / 16 * 256;

    for (int ax1_0_2 = 0; ax1_0_2 < 2; ++ax1_0_2)
    {
      for (int local_id = 0; local_id < 8; ++local_id)
      {

        C_ptr_local[ax1_0_2 * 16 + (((local_id / 2) % 2) * 8) * N +
                    (local_id % 2) + (local_id / 4) * 8] =
            sycl::ext::intel::math::float2half_rn(
                C_warp[((ax0_0 * 16) + (ax1_0_2 * 8)) + local_id]);
      };
    }
  }
}


// conv_backward_cuda_m16n16k64_m16n16k64_m16n16k16_tf32tf32f32
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
/*
DPCT1110:34: The total declared local variable size in device function
conv_backward_cuda_setting1_mode1_tf32tf32f32 exceeds 128 bytes and may cause
high register pressure. Consult with your hardware vendor to find the total
register size available and adjust the code, or use smaller sub-group size to
avoid high register pressure.
*/
void conv_backward_cuda_setting1_mode1_tf32tf32f32(
    int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters,
    int split_mask_len, int reduced_mask_len, int reorder_loc_len,
    float *__restrict__ A, float *__restrict__ B,
    int *__restrict__ reduced_mask, int *__restrict__ out_in_map,
    int *__restrict__ reorder_loc, float *__restrict__ C)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int K_tile = 16;
  int K_tile_padded = K_tile * ((K_original + K_tile - 1) / K_tile);

  float C_warp[8];
  auto &A_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2560]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  auto &B_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2560]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  float A_shared_warp[8];
  float B_shared_warp[8];
  for (int i = 0; i < 8; ++i)
  {
    C_warp[0 + i] = 0.0;
  }

  // hoisting shared pointer offsets
  int j_factors1 = (N + 15) / 16 / 1;
  int blockIdx_x = 0;
  int blockIdx_y = item_ct1.get_group(2) %
                   ((K_original + 15) / 16 * kernel_volume * j_factors1);
  int blockIdx_z = item_ct1.get_group(2) /
                   ((K_original + 15) / 16 * kernel_volume * j_factors1);
  float *cur_C = C + blockIdx_z * kernel_volume * K_original * N;
  int *out_in_map_ptr =
      out_in_map +
      (item_ct1.get_local_id(1) * 16 + item_ct1.get_local_id(2) / 2) *
          kernel_volume +
      ((item_ct1.get_local_id(1) * 256) % 16) / K_tile_padded +
      ((item_ct1.get_local_id(2) * 8) % 16) / K_tile_padded +
      (blockIdx_y / j_factors1 * 16) / K_tile_padded;
  float *A_ptr = A + ((item_ct1.get_local_id(1) * 256 % 16) % K_tile_padded) +
                 ((item_ct1.get_local_id(2) * 8 % 16) % K_tile_padded) +
                 ((blockIdx_y / j_factors1 * 16) % K_tile_padded);
  float *B_ptr =
      B + (blockIdx_y % j_factors1) * 16 + (item_ct1.get_local_id(2) * 8) % 16;
  int reorder_offset =
      item_ct1.get_local_id(1) * 256 / 16 + item_ct1.get_local_id(2) * 8 / 16;
  int K_iters = ((M_fwd + 63) / 64 + split_k_iters - 1) / split_k_iters;
  int kernel_offset = (blockIdx_y / j_factors1) / ((K_original + K_tile - 1) / K_tile);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_ptr = reorder_loc + split_mask_iter * reorder_loc_len;
  int* reduced_mask_ptr = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;
  int cur_C_ic_start = (blockIdx_y / j_factors1 * 16) % K_tile_padded +
                       (item_ct1.get_local_id(2) / 4);
  int cur_C_oc_start = (blockIdx_y % j_factors1) * 16 +
                       item_ct1.get_local_id(1) / 1 * 16 +
                       (item_ct1.get_local_id(2) % 4) * 2;
  float *C_ptr = cur_C + (kernel_offset * K_original + cur_C_ic_start) * N + cur_C_oc_start;

  int A_pred_guard = 0;
  int B_pred_guard = 0;
  if constexpr (K_ld_check)
  {
    int A_ld_start = ((item_ct1.get_local_id(1) * 256 % 16) % K_tile_padded) +
                     ((item_ct1.get_local_id(2) * 8 % 16) % K_tile_padded) +
                     ((blockIdx_y / j_factors1 * 16) % K_tile_padded);
    int A_ld_amount = sycl::min(A_ld_start + 8, K_original) - A_ld_start;
    int A_ld_bound = A_ld_amount / (K_ld_factor / 4);

    for (int i = 0; i < A_ld_bound; i++)
      A_pred_guard |= (1 << i);
  }
  else
    // load twice
    A_pred_guard = 3;
  if constexpr (N_ld_check)
  {
    int B_ld_start =
        (blockIdx_y % j_factors1) * 16 + (item_ct1.get_local_id(2) * 8) % 16;
    int B_ld_amount = sycl::min(B_ld_start + 8, N) - B_ld_start;
    int B_ld_bound = B_ld_amount / (N_ld_factor / 4);

    for (int i = 0; i < B_ld_bound; i++)
      B_pred_guard |= (1 << i);
  }
  else
    // load twice
    B_pred_guard = 3;


  for (int _i2_0_0 = 0; _i2_0_0 < K_iters - 1; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;
    
    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    float* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    /*
    DPCT1118:35: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume 
        + (ax0_ax1_fused_0 * 256 % 16) / K_tile_padded
      ];

      if (input_idx != -1)
      {
        sycl::uint4 A_loaded[2] = {sycl::uint4(0, 0, 0, 0)};
        global_load<K_ld_factor>(A_loaded[0], A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded), A_pred_guard);
        global_load<K_ld_factor>(A_loaded[1], A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded) + 4, A_pred_guard >> (4 * 4 / K_ld_factor));
        *(sycl::ulong4 *)(A_shared +
                          (((ax0_ax1_fused_0 * 640) +
                            ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                           ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            *reinterpret_cast<sycl::ulong4 *>(A_loaded);
      }
      else
      {
        *(sycl::ulong4 *)(A_shared +
                          (((ax0_ax1_fused_0 * 640) +
                            ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                           ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            sycl::ulong4(0ULL, 0ULL, 0ULL, 0ULL);
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
    {
  
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 16;
      int v0 = reorder_loc_ptr[reorder_offset_inner];
      sycl::uint4 B_loaded[2] = {sycl::uint4(0, 0, 0, 0)};
      global_load<N_ld_factor>(B_loaded[0], B_ptr + v0 * N, B_pred_guard);
      global_load<N_ld_factor>(B_loaded[1], B_ptr + v0 * N + 4, B_pred_guard >> (4 * 4 / N_ld_factor));
      *(sycl::ulong4 *)(B_shared +
                        (((ax0_ax1_fused_0_1 * 640) +
                          ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                         ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
          *reinterpret_cast<sycl::ulong4 *>(B_loaded);
    }

    /*
    DPCT1118:36: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int local_size = 0; local_size < 8; ++local_size)
      {
        A_shared_warp[local_size] =
            A_shared[(((((i2_0_1 * 640) + ((local_size >> 1) * 160)) +
                        ((((int)item_ct1.get_local_id(2)) & 3) * 40)) +
                       ((local_size & 1) * 8)) +
                      (((int)item_ct1.get_local_id(2)) >> 2))];
      }
      for (int local_size_1 = 0; local_size_1 < 8; ++local_size_1)
      {
        B_shared_warp[local_size_1] =
            B_shared[(((((i2_0_1 * 640) + ((local_size_1 & 3) * 160)) +
                        ((((int)item_ct1.get_local_id(2)) & 3) * 40)) +
                       ((local_size_1 >> 2) * 8)) +
                      (((int)item_ct1.get_local_id(2)) >> 2))];
      }
#if DPCT_COMPATIBILITY_TEMP >= 800
      {
        /*
        DPCT1053:37: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 0))[0]),
              "=f"(((float *)(C_warp + 0))[1]),
              "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]),
              "r"(((unsigned *)(A_shared_warp + 0))[1]),
              "r"(((unsigned *)(A_shared_warp + 0))[2]),
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + 0))[0]),
              "r"(((unsigned *)(B_shared_warp + 0))[1]),
              "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]),
              "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        /*
        DPCT1053:38: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 4))[0]),
              "=f"(((float *)(C_warp + 4))[1]),
              "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]),
              "r"(((unsigned *)(A_shared_warp + 0))[1]),
              "r"(((unsigned *)(A_shared_warp + 0))[2]),
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + 4))[0]),
              "r"(((unsigned *)(B_shared_warp + 4))[1]),
              "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]),
              "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }

      {
        /*
        DPCT1053:39: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 0))[0]),
              "=f"(((float *)(C_warp + 0))[1]),
              "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 4))[0]),
              "r"(((unsigned *)(A_shared_warp + 4))[1]),
              "r"(((unsigned *)(A_shared_warp + 4))[2]),
              "r"(((unsigned *)(A_shared_warp + 4))[3]),
              "r"(((unsigned *)(B_shared_warp + 2))[0]),
              "r"(((unsigned *)(B_shared_warp + 2))[1]),
              "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]),
              "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        /*
        DPCT1053:40: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 4))[0]),
              "=f"(((float *)(C_warp + 4))[1]),
              "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 4))[0]),
              "r"(((unsigned *)(A_shared_warp + 4))[1]),
              "r"(((unsigned *)(A_shared_warp + 4))[2]),
              "r"(((unsigned *)(A_shared_warp + 4))[3]),
              "r"(((unsigned *)(B_shared_warp + 6))[0]),
              "r"(((unsigned *)(B_shared_warp + 6))[1]),
              "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]),
              "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif 
    }
  }
  
  for (int _i2_0_0 = K_iters - 1; _i2_0_0 < K_iters; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * (K_iters - 1);
    if (i2_0_0 >= (M_fwd + 63) / 64)
      continue;
  
    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    float* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    /*
    DPCT1118:41: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume 
        + (ax0_ax1_fused_0 * 256 % 16) / K_tile_padded
      ];

      if (input_idx != -1)
      {
        sycl::uint4 A_loaded[2] = {sycl::uint4(0, 0, 0, 0)};
        global_load<K_ld_factor>(A_loaded[0], A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded), A_pred_guard);
        global_load<K_ld_factor>(A_loaded[1], A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded) + 4, A_pred_guard >> (4 * 4 / K_ld_factor));
        *(sycl::ulong4 *)(A_shared +
                          (((ax0_ax1_fused_0 * 640) +
                            ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                           ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            *reinterpret_cast<sycl::ulong4 *>(A_loaded);
      }
      else
      {
        *(sycl::ulong4 *)(A_shared +
                          (((ax0_ax1_fused_0 * 640) +
                            ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                           ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            sycl::ulong4(0ULL, 0ULL, 0ULL, 0ULL);
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
    {
  
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 16;
      if (reorder_offset_inner < M_fwd){
        int v0 = reorder_loc_ptr[reorder_offset_inner];
        sycl::uint4 B_loaded[2] = {sycl::uint4(0, 0, 0, 0)};
        global_load<N_ld_factor>(B_loaded[0], B_ptr + v0 * N, B_pred_guard);
        global_load<N_ld_factor>(B_loaded[1], B_ptr + v0 * N + 4, B_pred_guard >> (4 * 4 / N_ld_factor));
        *(sycl::ulong4 *)(B_shared +
                          (((ax0_ax1_fused_0_1 * 640) +
                            ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                           ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            *reinterpret_cast<sycl::ulong4 *>(B_loaded);
      }
      else 
      {
        *(sycl::ulong4 *)(B_shared +
                          (((ax0_ax1_fused_0_1 * 640) +
                            ((((int)item_ct1.get_local_id(2)) >> 1) * 40)) +
                           ((((int)item_ct1.get_local_id(2)) & 1) * 8))) =
            sycl::ulong4(0, 0, 0, 0);
      }
    }

    /*
    DPCT1118:42: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int local_size = 0; local_size < 8; ++local_size)
      {
        A_shared_warp[local_size] =
            A_shared[(((((i2_0_1 * 640) + ((local_size >> 1) * 160)) +
                        ((((int)item_ct1.get_local_id(2)) & 3) * 40)) +
                       ((local_size & 1) * 8)) +
                      (((int)item_ct1.get_local_id(2)) >> 2))];
      }
      for (int local_size_1 = 0; local_size_1 < 8; ++local_size_1)
      {
        B_shared_warp[local_size_1] =
            B_shared[(((((i2_0_1 * 640) + ((local_size_1 & 3) * 160)) +
                        ((((int)item_ct1.get_local_id(2)) & 3) * 40)) +
                       ((local_size_1 >> 2) * 8)) +
                      (((int)item_ct1.get_local_id(2)) >> 2))];
      }
#if DPCT_COMPATIBILITY_TEMP >= 800
      {
        /*
        DPCT1053:43: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 0))[0]),
              "=f"(((float *)(C_warp + 0))[1]),
              "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]),
              "r"(((unsigned *)(A_shared_warp + 0))[1]),
              "r"(((unsigned *)(A_shared_warp + 0))[2]),
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + 0))[0]),
              "r"(((unsigned *)(B_shared_warp + 0))[1]),
              "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]),
              "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        /*
        DPCT1053:44: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 4))[0]),
              "=f"(((float *)(C_warp + 4))[1]),
              "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]),
              "r"(((unsigned *)(A_shared_warp + 0))[1]),
              "r"(((unsigned *)(A_shared_warp + 0))[2]),
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + 4))[0]),
              "r"(((unsigned *)(B_shared_warp + 4))[1]),
              "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]),
              "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }

      {
        /*
        DPCT1053:45: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 0))[0]),
              "=f"(((float *)(C_warp + 0))[1]),
              "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 4))[0]),
              "r"(((unsigned *)(A_shared_warp + 4))[1]),
              "r"(((unsigned *)(A_shared_warp + 4))[2]),
              "r"(((unsigned *)(A_shared_warp + 4))[3]),
              "r"(((unsigned *)(B_shared_warp + 2))[0]),
              "r"(((unsigned *)(B_shared_warp + 2))[1]),
              "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]),
              "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        /*
        DPCT1053:46: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};"
            : "=f"(((float *)(C_warp + 4))[0]),
              "=f"(((float *)(C_warp + 4))[1]),
              "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 4))[0]),
              "r"(((unsigned *)(A_shared_warp + 4))[1]),
              "r"(((unsigned *)(A_shared_warp + 4))[2]),
              "r"(((unsigned *)(A_shared_warp + 4))[3]),
              "r"(((unsigned *)(B_shared_warp + 6))[0]),
              "r"(((unsigned *)(B_shared_warp + 6))[1]),
              "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]),
              "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif 
    }
  }
  
  for (int local_id = 0; local_id < 8; ++local_id)
  {
    if constexpr (K_ld_check || N_ld_check)
    {
      if (cur_C_ic_start + ((local_id / 2) % 2) * 8 < K_original && cur_C_oc_start + (local_id % 2) + (local_id / 4) * 8 < N)
        C_ptr[+(((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = C_warp[0 + local_id];
    }
    else
    {
      C_ptr[+(((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = C_warp[0 + local_id];
    }
  }
}


// conv_backward_cuda_m32n64k64_m32n32k64_m16n16k16_tf32tf32f32
/*
DPCT1110:47: The total declared local variable size in device function
conv_backward_cuda_setting2_mode1_tf32tf32f32 exceeds 128 bytes and may cause
high register pressure. Consult with your hardware vendor to find the total
register size available and adjust the code, or use smaller sub-group size to
avoid high register pressure.
*/
void conv_backward_cuda_setting2_mode1_tf32tf32f32(
    int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters,
    int split_mask_len, int reduced_mask_len, int reorder_loc_len,
    float *__restrict__ A, float *__restrict__ B,
    int *__restrict__ reduced_mask, int *__restrict__ out_in_map,
    int *__restrict__ reorder_loc, float *__restrict__ C)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  float C_warp[32];
  auto &A_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2560]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  auto &B_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4608]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  float A_shared_warp[16];
  float B_shared_warp[16];
  for (int i0_0_3_init = 0; i0_0_3_init < 2; ++i0_0_3_init)
  {
    for (int i1_0_4_init = 0; i1_0_4_init < 2; ++i1_0_4_init)
    {
      for (int i = 0; i < 8; ++i)
      {
        C_warp[((i0_0_3_init * 16) + (i1_0_4_init * 8)) + i] = 0.0;
      };
    }
  }

  // hoisting shared pointer offsets
  int j_factors1 = N / 16 / 4;
  int blockIdx_x = 0;
  int blockIdx_y = item_ct1.get_group(2) %
                   ((K_original * kernel_volume + 31) / 32 * j_factors1);
  int blockIdx_z = item_ct1.get_group(2) /
                   ((K_original * kernel_volume + 31) / 32 * j_factors1);
  float *cur_C = C + blockIdx_z * kernel_volume * N * K_original;
  int *out_in_map_ptr =
      out_in_map +
      (item_ct1.get_local_id(1) * 8 + item_ct1.get_local_id(2) / 4) *
          kernel_volume +
      ((item_ct1.get_local_id(1) * 256) % 32) / K_original +
      ((item_ct1.get_local_id(2) * 8) % 32) / K_original +
      (blockIdx_y / j_factors1 * 32) / K_original;
  float *A_ptr = A + ((item_ct1.get_local_id(1) * 256 % 32) % K_original) +
                 ((item_ct1.get_local_id(2) * 8 % 32) % K_original) +
                 ((blockIdx_y / j_factors1 * 32) % K_original);
  float *B_ptr =
      B + (blockIdx_y % j_factors1) * 64 + (item_ct1.get_local_id(2) * 8) % 64;
  int reorder_offset =
      item_ct1.get_local_id(1) * 256 / 64 + item_ct1.get_local_id(2) * 8 / 64;
  float *C_ptr =
      cur_C + blockIdx_x / 1 * 108 * N / 16 * 256 +
      blockIdx_y / j_factors1 * 2 * N / 16 * 256 +
      (item_ct1.get_local_id(1) % 1) * 2 * N / 16 * 256 +
      (blockIdx_x % 1) * j_factors1 * 64 + (blockIdx_y % j_factors1) * 64 +
      item_ct1.get_local_id(1) / 1 * 32 + (item_ct1.get_local_id(2) % 4) * 2 +
      (item_ct1.get_local_id(2) / 4) * N;
  int K_iters = ((M_fwd + 63) / 64 + split_k_iters - 1) / split_k_iters;
  int kernel_offset = (blockIdx_y / j_factors1) / (K_original / 32);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_ptr = reorder_loc + split_mask_iter * reorder_loc_len;
  int* reduced_mask_ptr = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;

  for (int _i2_0_0 = 0; _i2_0_0 < K_iters - 1; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;
    
    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    float* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    /*
    DPCT1118:48: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume 
        + (ax0_ax1_fused_0 * 512 % 32) / K_original
      ];

      if (input_idx != -1)
      {
        *(sycl::ulong4 *)(A_shared +
                          ((((ax0_ax1_fused_0 * 640) +
                             (((int)item_ct1.get_local_id(1)) * 320)) +
                            ((((int)item_ct1.get_local_id(2)) >> 2) * 40)) +
                           ((((int)item_ct1.get_local_id(2)) & 3) * 8))) =
            *(sycl::ulong4 *)(A_ptr_local + input_idx * K_original +
                              ((ax0_ax1_fused_0 * 512 % 32) % K_original));
      }
      else
      {
        *(sycl::ulong4 *)(A_shared +
                          ((((ax0_ax1_fused_0 * 640) +
                             (((int)item_ct1.get_local_id(1)) * 320)) +
                            ((((int)item_ct1.get_local_id(2)) >> 2) * 40)) +
                           ((((int)item_ct1.get_local_id(2)) & 3) * 8))) =
            sycl::ulong4(0ULL, 0ULL, 0ULL, 0ULL);
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
  
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 8;
      int v0 = reorder_loc_ptr[reorder_offset_inner];
      *(sycl::ulong4 *)(B_shared +
                        ((((ax0_ax1_fused_0_1 * 576) +
                           (((int)item_ct1.get_local_id(1)) * 288)) +
                          ((((int)item_ct1.get_local_id(2)) >> 3) * 72)) +
                         ((((int)item_ct1.get_local_id(2)) & 7) * 8))) =
          *(sycl::ulong4 *)(B_ptr + v0 * N);
    }

    /*
    DPCT1118:49: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {
        for (int local_size = 0; local_size < 8; ++local_size)
        {
          A_shared_warp[((ax1_0 * 8) + local_size)] =
              A_shared[((((((i2_0_1 * 640) + ((local_size >> 1) * 160)) +
                           ((((int)item_ct1.get_local_id(2)) & 3) * 40)) +
                          (ax1_0 * 16)) +
                         ((local_size & 1) * 8)) +
                        (((int)item_ct1.get_local_id(2)) >> 2))];
        }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
      {
        for (int local_size_1 = 0; local_size_1 < 8; ++local_size_1)
        {
          B_shared_warp[((ax1_0_1 * 8) + local_size_1)] =
              B_shared[(((((((i2_0_1 * 1152) + ((local_size_1 & 3) * 288)) +
                            ((((int)item_ct1.get_local_id(2)) & 3) * 72)) +
                           (((int)item_ct1.get_local_id(1)) * 32)) +
                          (ax1_0_1 * 16)) +
                         ((local_size_1 >> 2) * 8)) +
                        (((int)item_ct1.get_local_id(2)) >> 2))];
        }
      }

      for (int i0_0_3 = 0; i0_0_3 < 2; ++i0_0_3)
      {
        for (int i1_0_4 = 0; i1_0_4 < 2; ++i1_0_4)
        {
#if DPCT_COMPATIBILITY_TEMP >= 800
          {
            /*
            DPCT1053:50: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]),
                  "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]),
                  "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            /*
            DPCT1053:51: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }

          {
            /*
            DPCT1053:52: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[2]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[3]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            /*
            DPCT1053:53: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[2]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[3]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif 
        }
      }
    }
  }
  
  for (int _i2_0_0 = K_iters - 1; _i2_0_0 < K_iters; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * (K_iters - 1);
    if (i2_0_0 >= (M_fwd + 63) / 64)
      continue;
  
    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    float* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    /*
    DPCT1118:54: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume 
        + (ax0_ax1_fused_0 * 512 % 32) / K_original
      ];

      if (input_idx != -1)
      {
        *(sycl::ulong4 *)(A_shared +
                          ((((ax0_ax1_fused_0 * 640) +
                             (((int)item_ct1.get_local_id(1)) * 320)) +
                            ((((int)item_ct1.get_local_id(2)) >> 2) * 40)) +
                           ((((int)item_ct1.get_local_id(2)) & 3) * 8))) =
            *(sycl::ulong4 *)(A_ptr_local + input_idx * K_original +
                              ((ax0_ax1_fused_0 * 512 % 32) % K_original));
      }
      else
      {
        *(sycl::ulong4 *)(A_shared +
                          ((((ax0_ax1_fused_0 * 640) +
                             (((int)item_ct1.get_local_id(1)) * 320)) +
                            ((((int)item_ct1.get_local_id(2)) >> 2) * 40)) +
                           ((((int)item_ct1.get_local_id(2)) & 3) * 8))) =
            sycl::ulong4(0ULL, 0ULL, 0ULL, 0ULL);
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
  
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 8;
      if (reorder_offset_inner < M_fwd){
        int v0 = reorder_loc_ptr[reorder_offset_inner];
        *(sycl::ulong4 *)(B_shared +
                          ((((ax0_ax1_fused_0_1 * 576) +
                             (((int)item_ct1.get_local_id(1)) * 288)) +
                            ((((int)item_ct1.get_local_id(2)) >> 3) * 72)) +
                           ((((int)item_ct1.get_local_id(2)) & 7) * 8))) =
            *(sycl::ulong4 *)(B_ptr + v0 * N);
        }
        else 
        {
          *(sycl::ulong4 *)(B_shared +
                            ((((ax0_ax1_fused_0_1 * 576) +
                               (((int)item_ct1.get_local_id(1)) * 288)) +
                              ((((int)item_ct1.get_local_id(2)) >> 3) * 72)) +
                             ((((int)item_ct1.get_local_id(2)) & 7) * 8))) =
              sycl::ulong4(0ULL, 0ULL, 0ULL, 0ULL);
        }
      }

    /*
    DPCT1118:55: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {
        for (int local_size = 0; local_size < 8; ++local_size)
        {
          A_shared_warp[((ax1_0 * 8) + local_size)] =
              A_shared[((((((i2_0_1 * 640) + ((local_size >> 1) * 160)) +
                           ((((int)item_ct1.get_local_id(2)) & 3) * 40)) +
                          (ax1_0 * 16)) +
                         ((local_size & 1) * 8)) +
                        (((int)item_ct1.get_local_id(2)) >> 2))];
        }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
      {
        for (int local_size_1 = 0; local_size_1 < 8; ++local_size_1)
        {
          B_shared_warp[((ax1_0_1 * 8) + local_size_1)] =
              B_shared[(((((((i2_0_1 * 1152) + ((local_size_1 & 3) * 288)) +
                            ((((int)item_ct1.get_local_id(2)) & 3) * 72)) +
                           (((int)item_ct1.get_local_id(1)) * 32)) +
                          (ax1_0_1 * 16)) +
                         ((local_size_1 >> 2) * 8)) +
                        (((int)item_ct1.get_local_id(2)) >> 2))];
        }
      }

      for (int i0_0_3 = 0; i0_0_3 < 2; ++i0_0_3)
      {
        for (int i1_0_4 = 0; i1_0_4 < 2; ++i1_0_4)
        {
#if DPCT_COMPATIBILITY_TEMP >= 800
          {
            /*
            DPCT1053:56: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]),
                  "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]),
                  "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            /*
            DPCT1053:57: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]),
                  "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }

          {
            /*
            DPCT1053:58: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[2]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[3]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]),
                  "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            /*
            DPCT1053:59: Migration of device assembly code is not supported.
            */
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
                "%13};"
                : "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "=f"(((float *)(C_warp +
                                  (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[2]),
                  "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[3]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]),
                  "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]),
                  "f"(((float *)(C_warp +
                                 (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif 
        }
      }
    }
  }
  
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0)
  {

    float* C_ptr_local = C_ptr + ax0_0 * N / 16 * 256;

    for (int ax1_0_2 = 0; ax1_0_2 < 2; ++ax1_0_2)
    {
      for (int local_id = 0; local_id < 8; ++local_id)
      {

        C_ptr_local[ax1_0_2 * 16 + (((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = C_warp[((ax0_0 * 16) + (ax1_0_2 * 8)) + local_id];
      };
    }
  }
}


// conv_backward_cuda_m16n16k64_f32f32f32_sort
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
/*
DPCT1110:60: The total declared local variable size in device function
conv_backward_cuda_setting1_mode1_f32f32f32 exceeds 128 bytes and may cause high
register pressure. Consult with your hardware vendor to find the total register
size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void conv_backward_cuda_setting1_mode1_f32f32f32(
    int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters,
    int split_mask_len, int reduced_mask_len, int reorder_loc_len,
    float *__restrict__ A, float *__restrict__ B,
    int *__restrict__ reduced_mask, int *__restrict__ out_in_map,
    int *__restrict__ reorder_loc, float *__restrict__ C)
{

  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int j_factors1 = (N + 15) / 16;
  int blockIdx_x = 0;
  int blockIdx_y = item_ct1.get_group(2) %
                   ((K_original + 15) / 16 * kernel_volume * j_factors1);
  int blockIdx_z = item_ct1.get_group(2) /
                   ((K_original + 15) / 16 * kernel_volume * j_factors1);

  const int K_tile = 16;
  int K_tile_padded = K_tile * ((K_original + K_tile - 1) / K_tile);

  float C_local[8];
  auto &A_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  auto &B_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());

#pragma unroll
  for (int i = 0; i < 8; ++i)   
  {
    C_local[i] = 0.0;
  }
  
  int blockIdx_m = blockIdx_y / j_factors1;
  int blockIdx_n = blockIdx_y % j_factors1;
  int threadIdx_x = (int)item_ct1.get_local_id(2);

  int kernel_offset = blockIdx_m / (K_tile_padded / 16);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_local = reorder_loc + split_mask_iter * reorder_loc_len ;
  int* reduced_mask_local = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;
  
  int channel_offset = (blockIdx_m * 16 + ((threadIdx_x * 4) % 16)) % K_tile_padded;
  int K_loops = ((M_fwd + 63 ) / 64 + split_k_iters - 1) / split_k_iters;

  // hoisting shared pointer offsets
  int * out_in_map_ptr = out_in_map
                          + (threadIdx_x / (16/4)) * kernel_volume
                          + kernel_offset;
  float * A_ptr = A + channel_offset;

  // reorder is performed on B's rows.
  float * B_ptr = B
                    + (blockIdx_n * 16) + ((threadIdx_x * 4) % 16); 
  int reorder_offset = threadIdx_x /(16/4);

  float * A_shared_ptr = A_shared + (threadIdx_x * 4);
  float * B_shared_ptr = B_shared + (threadIdx_x * 4);

  float * A_shared_reduce_ptr =  A_shared + (threadIdx_x / 4); 
  float * B_shared_reduce_ptr = B_shared + (threadIdx_x % 4);

  // splitK offset
  float * cur_C = C + blockIdx_z * K_original * kernel_volume * N;
  int cur_C_ic_start = (blockIdx_m * 16 + (threadIdx_x / 4)) % K_tile_padded;
  int cur_C_oc_start = blockIdx_n * 16 + (threadIdx_x % 4);
  float * C_ptr = cur_C + (kernel_offset * K_original + cur_C_ic_start) * N + cur_C_oc_start; 

  int A_pred_guard = 0;
  int B_pred_guard = 0;
  if constexpr (K_ld_check) // IC % cta_M != 0 
  {
    int A_ld_start = channel_offset;
    int A_ld_amount = sycl::min(A_ld_start + 4, K_original) - A_ld_start;
    int A_ld_bound = A_ld_amount / (K_ld_factor / 4);

    for (int i = 0; i < A_ld_bound; i++)
      A_pred_guard |= (1 << i);
  }
  else
    A_pred_guard = 1;

  if constexpr (N_ld_check) // OC % cta_N != 0
  {
    int B_ld_start = (blockIdx_n * 16) + ((threadIdx_x * 4) % 16);
    int B_ld_amount = sycl::min(B_ld_start + 4, N) - B_ld_start;
    int B_ld_bound = B_ld_amount / (N_ld_factor / 4);

    for (int i = 0; i < B_ld_bound; i++)
      B_pred_guard |= (1 << i);
  }
  else
    B_pred_guard = 1;

  #pragma unroll
  for (int _k_0 = 0; _k_0 < K_loops - 1; ++_k_0) 
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    bool bit_flag = (bool)(reduced_mask_local[k_0] & (1 << bitmask_shift));
    if (!bit_flag)
      continue;

    /*
    DPCT1118:61: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = out_in_map_ptr_local[(ax0_ax1_fused_0 *8) * kernel_volume]; 
      if (input_idx != -1)
      {
        // *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) =  // ax0_ax1_fused_0 * elements loaded in each loop
        //     *(float4*)(A_ptr + (input_idx * K_original));
        sycl::uint4 A_loaded = sycl::uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr + (input_idx * K_original) , A_pred_guard);
        *(sycl::uint4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) = A_loaded;
      }
      else 
      {
        *(sycl::uint4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) =
            sycl::uint4(0, 0, 0, 0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 8);
      int v0 = reorder_loc_local[reorder_offset_inner];
      //*(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) = 
      //    *(float4*)(B_ptr + v0 * N);
      sycl::uint4 B_loaded = sycl::uint4(0, 0, 0, 0);
      global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
      *(sycl::uint4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) = B_loaded;
    }

    /*
    DPCT1118:62: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1) 
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2) 
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
          C_local[i] = C_local[i] + 
                          A_shared_reduce_ptr[(vk_in_block * 16) + ((i / 4) * 8)]
                          * B_shared_reduce_ptr[(vk_in_block * 16) + ((i % 4) * 4)];
        }

      }
    }
  }
  for (int _k_0 = K_loops - 1; _k_0 < K_loops; ++_k_0) 
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    if (k_0 >= (M_fwd + 63) / 64)
      break;

    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    bool bit_flag = (bool)(reduced_mask_local[k_0] & (1 << bitmask_shift));
    if (!bit_flag)
      continue;

    /*
    DPCT1118:63: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = *(out_in_map_ptr_local + (ax0_ax1_fused_0 *8) * kernel_volume); 
      if (input_idx != -1)
      {
        // *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) =  // ax0_ax1_fused_0 * elements loaded in each loop
        //     *(float4*)(A_ptr + (input_idx * K_original));
        sycl::uint4 A_loaded = sycl::uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr + (input_idx * K_original) , A_pred_guard);
        *(sycl::uint4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) = A_loaded;
      }
      else 
      {
        *(sycl::uint4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) =
            sycl::uint4(0, 0, 0, 0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 8);
      if (reorder_offset_inner < M_fwd)
      {
        int v0 = reorder_loc_local[reorder_offset_inner];
        //*(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) = 
        //    *(float4*)(B_ptr + v0 * N);
        sycl::uint4 B_loaded = sycl::uint4(0, 0, 0, 0);
        global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
        *(sycl::uint4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) = B_loaded;
      }
      else
      {
        *(sycl::uint4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) =
            sycl::uint4(0, 0, 0, 0);
      }
    }

    /*
    DPCT1118:64: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1) 
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2) 
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
          C_local[i] = C_local[i] + 
                          A_shared_reduce_ptr[(vk_in_block * 16) + ((i / 4) * 8)]
                          * B_shared_reduce_ptr[(vk_in_block * 16) + ((i % 4) * 4)];
        }

      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i)
  {
    int local_row = ((i / 4) * 8);
    int local_col = ((i % 4) * 4); 
    if constexpr (K_ld_check || N_ld_check)
    {
      if ( ((cur_C_ic_start + local_row) < K_original) && ((cur_C_oc_start + local_col) < N) )
        C_ptr[local_row * N + local_col] = C_local[i];
        
    }
    else
    {
      C_ptr[local_row * N + local_col] = C_local[i];
    }
  }
}


// conv_backward_cuda_m32n64k64_f32f32f32_sort
/*
DPCT1110:65: The total declared local variable size in device function
conv_backward_cuda_setting2_mode1_f32f32f32 exceeds 128 bytes and may cause high
register pressure. Consult with your hardware vendor to find the total register
size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void conv_backward_cuda_setting2_mode1_f32f32f32(
    int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters,
    int split_mask_len, int reduced_mask_len, int reorder_loc_len,
    float *__restrict__ A, float *__restrict__ B,
    int *__restrict__ reduced_mask, int *__restrict__ out_in_map,
    int *__restrict__ reorder_loc, float *__restrict__ C)
{

  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int j_factors1 = (N + 63) / 64;
  int blockIdx_x = 0;
  int blockIdx_y = item_ct1.get_group(2) %
                   ((K_original * kernel_volume + 31) / 32 * j_factors1);
  int blockIdx_z = item_ct1.get_group(2) /
                   ((K_original * kernel_volume + 31) / 32 * j_factors1);

  float C_local[32];
  auto &A_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2048]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  auto &B_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4096]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());

#pragma unroll
  for (int i = 0; i < 32; ++i)   
  {
    C_local[i] = 0.0;
  }
  
  int blockIdx_m = blockIdx_y / j_factors1;
  int blockIdx_n = blockIdx_y % j_factors1;
  int threadIdx_x = (int)item_ct1.get_local_id(2);

  int kernel_offset = blockIdx_m / (K_original / 32);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_local = reorder_loc + split_mask_iter * reorder_loc_len;
  int* reduced_mask_local = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;

  int channel_offset = (blockIdx_m * 32 + ((threadIdx_x * 4) % 32)) % K_original;
  int K_loops = ((M_fwd + 63 ) / 64 + split_k_iters - 1) / split_k_iters;

  // hoisting shared pointer offsets
  int * out_in_map_ptr = out_in_map
                          + (threadIdx_x / (32/4)) * kernel_volume
                          + kernel_offset;
  float * A_ptr = A + channel_offset;

  // reorder is performed on B's rows.
  float * B_ptr = B
                    + (blockIdx_n * 64) + ((threadIdx_x * 4) % 64); 
  int reorder_offset = threadIdx_x /(64/4);

  float * A_shared_ptr = A_shared + (threadIdx_x * 4);
  float * B_shared_ptr = B_shared + (threadIdx_x * 4);

  float * A_shared_reduce_ptr =  A_shared + (threadIdx_x / 16); 
  float * B_shared_reduce_ptr = B_shared + (threadIdx_x % 16);

  // splitK offset
  float * cur_C = C + blockIdx_z * K_original * kernel_volume * N;
  int C_m_offset = blockIdx_m * 32 + (threadIdx_x / 16);  // C_m_offset
  int C_n_offset = blockIdx_n * 64  + (threadIdx_x % 16);
  // float * C_ptr = cur_C + C_m_offset * N + C_n_offset; 

  #pragma unroll
  for (int _k_0 = 0; _k_0 < K_loops - 1; ++_k_0) 
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    bool bit_flag = (bool)(reduced_mask_local[k_0] & (1 << bitmask_shift));
    if (!bit_flag)
      continue;

    /*
    DPCT1118:66: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = out_in_map_ptr_local[(ax0_ax1_fused_0 *8) * kernel_volume]; 
      if (input_idx != -1)
      {
        *(sycl::float4 *)(A_shared_ptr + (ax0_ax1_fused_0 *
                                          256)) = // ax0_ax1_fused_0 * elements
                                                  // loaded in each loop
            *(sycl::float4 *)(A_ptr + (input_idx * K_original));
      }
      else 
      {
        *(sycl::float4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) =
            sycl::float4(0.0, 0.0, 0.0, 0.0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 16; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 4);
      int v0 = reorder_loc_local[reorder_offset_inner];
      *(sycl::float4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) =
          *(sycl::float4 *)(B_ptr + v0 * N);
    }

    /*
    DPCT1118:67: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1) 
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2) 
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          C_local[i] = C_local[i] + 
                          A_shared_reduce_ptr[(vk_in_block * 32) + ((i / 4) * 4)]
                          * B_shared_reduce_ptr[(vk_in_block * 64) + ((i % 4) * 16)];
        }

      }
    }
  }
  for (int _k_0 = K_loops - 1; _k_0 < K_loops; ++_k_0) 
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    if (k_0 >= (M_fwd + 63) / 64)
      break;

    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    bool bit_flag = (bool)(reduced_mask_local[k_0] & (1 << bitmask_shift));
    if (!bit_flag)
      continue;

    /*
    DPCT1118:68: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = *(out_in_map_ptr_local + (ax0_ax1_fused_0 *8) * kernel_volume); 
      if (input_idx != -1)
      {
        *(sycl::float4 *)(A_shared_ptr + (ax0_ax1_fused_0 *
                                          256)) = // ax0_ax1_fused_0 * elements
                                                  // loaded in each loop
            *(sycl::float4 *)(A_ptr + (input_idx * K_original));
      }
      else 
      {
        *(sycl::float4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) =
            sycl::float4(0.0, 0.0, 0.0, 0.0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 16; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 4);
      if (reorder_offset_inner < M_fwd)
      {
        int v0 = reorder_loc_local[reorder_offset_inner];
        *(sycl::float4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) =
            *(sycl::float4 *)(B_ptr + v0 * N);
      }
      else
      {
        *(sycl::float4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) =
            sycl::float4(0.0, 0.0, 0.0, 0.0);
      }
    }

    /*
    DPCT1118:69: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1) 
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2) 
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          C_local[i] = C_local[i] + 
                          A_shared_reduce_ptr[(vk_in_block * 32) + ((i / 4) * 4)]
                          * B_shared_reduce_ptr[(vk_in_block * 64) + ((i % 4) * 16)];
        }

      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 32; ++i)
  {
      int C_m_offset_cur = C_m_offset + ((i / 4) * 4);
      int C_n_offset_cur = C_n_offset + ((i % 4) * 16); 
      cur_C[C_m_offset_cur * N + C_n_offset_cur] = C_local[i];
  }
}


at::Tensor conv_backward_wgrad_implicit_gemm_sorted_cuda(
    torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _out_in_map, torch::Tensor _reduced_mask,
    torch::Tensor _reorder_loc, const int split_k_iters,
    bool allow_tf32, bool allow_fp16)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  bool is_tf = allow_tf32;
  int num_in_feats = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);
  int kernel_volume = _out_in_map.size(1);
  int split_mask_num = _reorder_loc.size(0);
  int split_mask_len = (kernel_volume + split_mask_num - 1) / split_mask_num;
  int reduced_mask_len = _reduced_mask.size(1);
  int reorder_loc_len = _reorder_loc.size(1);  
  auto options =
      torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out_feats = torch::empty({split_k_iters, num_in_channels * kernel_volume, _kernel.size(1)}, options);
  int num_out_feats = _out_feats.size(1);
  int num_out_channels = _out_feats.size(2);
  auto reduced_mask = _reduced_mask.data_ptr<int>();
  auto out_in_map = _out_in_map.data_ptr<int>();
  auto reorder_loc = _reorder_loc.data_ptr<int>();
  bool is_half = _in_feats.scalar_type() == at::ScalarType::Half;

  if (is_half)
  {
    // throw std::runtime_error("FP16 kernels have not been updated for split mask implimentation.");
    if (!allow_fp16)
    {
      throw std::runtime_error("FP16 kernels are not supported for implicit GEMM now for SM75-.");
    }
    auto in_feats =
        reinterpret_cast<sycl::half *>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<sycl::half *>(_kernel.data_ptr<at::Half>());
    auto out_feats =
        reinterpret_cast<sycl::half *>(_out_feats.data_ptr<at::Half>());

    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int j_factors1 = num_out_channels / 64 / 1;
      dpct::dim3 num_blocks(1 * num_in_channels * kernel_volume / 32 *
                            j_factors1 * split_k_iters);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dpct::dim3 threads_per_block(32, 2);
      {
        dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
            .has_capability_or_fail({sycl::aspect::fp16});

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        q_ct1.submit([&](sycl::handler &cgh) {
          int _kernel_size_ct0 = _kernel.size(0);

          cgh.parallel_for(
              sycl::nd_range<3>(num_blocks * threads_per_block,
                                threads_per_block),
              exp_props, [=](sycl::nd_item<3> item_ct1) {
                conv_backward_cuda_setting2_mode1_f16f16f32(
                    _kernel_size_ct0, num_in_channels, num_out_channels,
                    kernel_volume, split_k_iters, split_mask_len,
                    reduced_mask_len, reorder_loc_len, in_feats, kernel,
                    reduced_mask, out_in_map, reorder_loc, out_feats);
              });
        });
      }
    }
    else
    {
      int j_factors1 = (num_out_channels + 15) / 16 / 1;
      dpct::dim3 num_blocks(1 * (num_in_channels + 15) / 16 * kernel_volume *
                            j_factors1 * split_k_iters);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dpct::dim3 threads_per_block(32, 1);
      // conv_backward_cuda_setting1_mode1_f16f16f32<<<num_blocks, threads_per_block>>>(
      //     _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:70: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<16, 16, false,
                                                              false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 8 == 0)
        {
          /*
          DPCT1049:71: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<16, 16, false,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:72: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<16, 8, false,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:73: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<16, 4, false,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:74: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<16, 2, false,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
      else if (num_in_channels % 8 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:75: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<16, 16, true,
                                                              false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 8 == 0)
        {
          /*
          DPCT1049:76: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<16, 16, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:77: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<16, 8, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:78: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<16, 4, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:79: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<16, 2, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:80: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<8, 16, true,
                                                              false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 8 == 0)
        {
          /*
          DPCT1049:81: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<8, 16, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:82: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<8, 8, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:83: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<8, 4, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:84: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<8, 2, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:85: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<4, 16, true,
                                                              false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 8 == 0)
        {
          /*
          DPCT1049:86: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<4, 16, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:87: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<4, 8, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:88: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<4, 4, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:89: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<4, 2, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:90: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<2, 16, true,
                                                              false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 8 == 0)
        {
          /*
          DPCT1049:91: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<2, 16, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:92: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<2, 8, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:93: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<2, 4, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:94: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          dpct::get_device(dpct::get_device_id(q_ct1.get_device()))
              .has_capability_or_fail({sycl::aspect::fp16});

          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f16f16f32<2, 2, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
    }
  }
  else if (is_tf)
  {
    //throw std::runtime_error("TF32 kernels have not been updated for split mask implimentation.");
    auto in_feats = _in_feats.data_ptr<float>();
    auto kernel = _kernel.data_ptr<float>();
    auto out_feats = _out_feats.data_ptr<float>();

    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int j_factors1 = num_out_channels / 64 / 1;
      dpct::dim3 num_blocks(1 * num_in_channels * kernel_volume / 32 *
                            j_factors1 * split_k_iters);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dpct::dim3 threads_per_block(32, 2);
      auto exp_props = sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::use_root_sync};
      q_ct1.submit([&](sycl::handler &cgh) {
        int _kernel_size_ct0 = _kernel.size(0);

        cgh.parallel_for(sycl::nd_range<3>(num_blocks * threads_per_block,
                                           threads_per_block),
                         exp_props, [=](sycl::nd_item<3> item_ct1) {
                           conv_backward_cuda_setting2_mode1_tf32tf32f32(
                               _kernel_size_ct0, num_in_channels,
                               num_out_channels, kernel_volume, split_k_iters,
                               split_mask_len, reduced_mask_len,
                               reorder_loc_len, in_feats, kernel, reduced_mask,
                               out_in_map, reorder_loc, out_feats);
                         });
      });
    }
    else
    {
      int j_factors1 = (num_out_channels + 15) / 16 / 1;
      dpct::dim3 num_blocks(1 * (num_in_channels + 15) / 16 * kernel_volume *
                            j_factors1 * split_k_iters);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dpct::dim3 threads_per_block(32, 1);
      // conv_backward_cuda_setting1_mode1_tf32tf32f32<<<num_blocks, threads_per_block>>>(
      //     _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:95: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 16, false,
                                                                false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:96: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 16, false,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:97: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 8, false,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:98: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 4, false,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:99: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 16, true,
                                                                false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:100: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 16, true,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:101: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 8, true,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:102: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 4, true,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:103: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<8, 16, true,
                                                                false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:104: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<8, 16, true,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:105: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<8, 8, true,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:106: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<8, 4, true,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:107: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<4, 16, true,
                                                                false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:108: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<4, 16, true,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:109: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<4, 8, true,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:110: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_tf32tf32f32<4, 4, true,
                                                                true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
    }
  }
  else // fp32fp32fp32
  {
    // printf("\nRun FP32 wgrad backward kernels!\n");
    auto in_feats = _in_feats.data_ptr<float>();
    auto kernel = _kernel.data_ptr<float>();
    auto out_feats = _out_feats.data_ptr<float>();

    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int block_num_M = (num_in_channels * kernel_volume) / 32;
      int block_num_N = (num_out_channels) / 64; //j_factors1

      dpct::dim3 num_blocks(block_num_M * block_num_N * split_k_iters);
      dpct::dim3 threads_per_block(64);
      auto exp_props = sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::use_root_sync};
      q_ct1.submit([&](sycl::handler &cgh) {
        int _kernel_size_ct0 = _kernel.size(0);

        cgh.parallel_for(sycl::nd_range<3>(num_blocks * threads_per_block,
                                           threads_per_block),
                         exp_props, [=](sycl::nd_item<3> item_ct1) {
                           conv_backward_cuda_setting2_mode1_f32f32f32(
                               _kernel_size_ct0, num_in_channels,
                               num_out_channels, kernel_volume, split_k_iters,
                               split_mask_len, reduced_mask_len,
                               reorder_loc_len, in_feats, kernel, reduced_mask,
                               out_in_map, reorder_loc, out_feats);
                         });
      });
    }
    else
    {
      int block_num_M = (num_in_channels + 15) / 16 * kernel_volume;
      int block_num_N = (num_out_channels - 1) / 16 + 1;

      dpct::dim3 num_blocks(block_num_M * block_num_N * split_k_iters);
      dpct::dim3 threads_per_block(32);
      // conv_backward_cuda_setting1_mode1_tf32tf32f32<<<num_blocks, threads_per_block>>>(
      //     _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:111: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<16, 16, false,
                                                              false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:112: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<16, 16, false,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:113: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<16, 8, false,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:114: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<16, 4, false,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:115: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<16, 16, true,
                                                              false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:116: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<16, 16, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:117: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<16, 8, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:118: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<16, 4, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:119: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<8, 16, true,
                                                              false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:120: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<8, 16, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:121: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<8, 8, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:122: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<8, 4, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          /*
          DPCT1049:123: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<4, 16, true,
                                                              false>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 4 == 0)
        {
          /*
          DPCT1049:124: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<4, 16, true,
                                                              true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else if (num_out_channels % 2 == 0)
        {
          /*
          DPCT1049:125: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<4, 8, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
        else
        {
          /*
          DPCT1049:126: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
          auto exp_props = sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::use_root_sync};
          q_ct1.submit([&](sycl::handler &cgh) {
            int _kernel_size_ct0 = _kernel.size(0);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * threads_per_block,
                                  threads_per_block),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                  conv_backward_cuda_setting1_mode1_f32f32f32<4, 4, true, true>(
                      _kernel_size_ct0, num_in_channels, num_out_channels,
                      kernel_volume, split_k_iters, split_mask_len,
                      reduced_mask_len, reorder_loc_len, in_feats, kernel,
                      reduced_mask, out_in_map, reorder_loc, out_feats);
                });
          });
        }
      }
    }
  }
  return _out_feats.sum(0);
}
