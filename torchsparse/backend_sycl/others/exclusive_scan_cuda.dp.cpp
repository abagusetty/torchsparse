#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/torch.h>
#include <torch/extension.h>

#include "exclusive_scan_cuda.h"

// to derive quantified address of activated features
void exclusive_scan_for_kernel_quantified(
                const int kv, 
                const int *input, 
                const int q, 
                // const int mid_kernel, 
                int *output,
                int *qoutput
                // bool precompute_mid
){
  // a thread for a scan
  const int id =
      sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) + 1;
  if (id >= kv){return;}
  int acc = 0;
  int qacc = 0;
#pragma unroll 
  for (int i = 0; i < id; i++){ 
    // if (precompute_mid && i == mid_kernel){continue;}
    acc += input[i];
    qacc += (input[i] + q - 1) / q * q;
  }
  output[id] = acc;
  qoutput[id] = qacc;
}

at::Tensor exclusive_scan_quantified_wrapper(
    const int k_vol, at::Tensor neighbor_offset, 
    at::Tensor neighbor_address, at::Tensor q_neighbor_address){

  int *knnz_ptr = neighbor_offset.data_ptr<int>();
  int *kpos_ptr = neighbor_address.data_ptr<int>();
  int *qkpos_ptr = q_neighbor_address.data_ptr<int>();

  /*
  DPCT1049:510: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto exp_props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    auto k_vol_ct0 = k_vol + 1;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, k_vol),
                                       sycl::range<3>(1, 1, k_vol)),
                     exp_props, [=](sycl::nd_item<3> item_ct1) {
                       exclusive_scan_for_kernel_quantified(
                           k_vol_ct0, knnz_ptr, 128, kpos_ptr, qkpos_ptr);
                     });
  });
  // We must have a tensor as return val for Pybind.
  at::Tensor status = at::zeros({1});
  return status;
}