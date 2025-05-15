#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/extension.h>
#include "reduce_bitmask_cuda.h"


// 1 block -- 4 warps -- 128 threads
// 1 warp -- 8 output elements (4 threads for 1 reduced element in int32)
// each thread reduce reduce_tile/4 int32 numbers
// threads in 1st warp finish the final reduction of 4 numbers

#define thd_per_blk 128
#define output_per_blk 32 // thd_per_blk / 4  -> (4 threads for 1 reduced element in int32)

extern "C" 
void reduce_mask_cuda_int32(
                                         int* __restrict__ bitmask, 
                                         int output_node_num,
                                         int reduced_row_num,
                                         int reduce_tile,
                                         int* __restrict__ reduced_bitmask) {

  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int split_mask_iter = item_ct1.get_group(1);
  int thread_size = reduce_tile / 4;
  int blockIdx_x = (int)item_ct1.get_group(2);
  int threadIdx_x = (int)item_ct1.get_local_id(2);
  int laneid = (threadIdx_x & 31);
  int warpid = (threadIdx_x >> 5);

  int bitmask_local = 0;
  auto &bitmask_shared =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<int[thd_per_blk]>(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
  int* final_reduce_ptr = bitmask_shared + (laneid << 2);

  int* bitmask_blk = bitmask + split_mask_iter * output_node_num;
  int* reduced_bitmask_blk = reduced_bitmask + split_mask_iter * reduced_row_num;
  int block_offset = blockIdx_x * thd_per_blk * thread_size;
  int thread_offset = block_offset + (threadIdx_x * thread_size);
  int load_len = sycl::min(thread_size, output_node_num - thread_offset);

#pragma unroll
  for (int i = 0; i < load_len; i++) {
    int load_offset = i + thread_offset;
    bitmask_local = bitmask_local | bitmask_blk[load_offset]; 
  }
  bitmask_shared[threadIdx_x] = bitmask_local;
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // final reduction
  if(warpid == 0){
    #pragma unroll
    for(int i = 1; i < 4; i++){
      final_reduce_ptr[0] = final_reduce_ptr[0] | final_reduce_ptr[i];
    }
    int output_offset = (blockIdx_x << 5) + laneid;
    if (output_offset < reduced_row_num){
      reduced_bitmask_blk[output_offset] = final_reduce_ptr[0];
    }
  }
}


torch::Tensor reduce_bitmask_cuda(
    torch::Tensor _bitmask_int,
    int M_tile
){
    if (M_tile % 4 != 0)
    {
      throw std::runtime_error("[Bitmask reduce] reduce tile size must be multiple of 4.");
    }
    int split_mask_num = _bitmask_int.size(0);
    int output_node_num = _bitmask_int.size(1);
    int reduced_row_num = (output_node_num - 1) / M_tile + 1;
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(_bitmask_int.device());
    torch::Tensor _reduced_bitmask_int = torch::zeros({split_mask_num, reduced_row_num}, options);

    auto bitmask_int = _bitmask_int.data_ptr<int>();
    auto reduced_bitmask_int = _reduced_bitmask_int.data_ptr<int>();

    dpct::dim3 num_blocks(((reduced_row_num - 1) / output_per_blk + 1),
                          split_mask_num);
    dpct::dim3 num_threads(thd_per_blk);

  auto exp_props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(num_blocks * num_threads, num_threads), exp_props,
      [=](sycl::nd_item<3> item_ct1) {
        reduce_mask_cuda_int32(bitmask_int, output_node_num, reduced_row_num,
                               M_tile, reduced_bitmask_int);
      });

    return _reduced_bitmask_int;
} 



