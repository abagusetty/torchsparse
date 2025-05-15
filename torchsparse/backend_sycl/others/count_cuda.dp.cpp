#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

// counting
// input N*3 int32 tensor output N*1 int64 tensor
void count_kernel(int N, const int *__restrict__ data,
                             int *__restrict__ out) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);
  if (i < N && data[i] >= 0) {
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &out[data[i]], 1);
  }
}

void count_wrapper(int N, const int *data, int *out) {
  /*
  DPCT1049:512: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto exp_props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, ceil((double)N / 512)) *
                            sycl::range<3>(1, 1, 512),
                        sycl::range<3>(1, 1, 512)),
      exp_props, [=](sycl::nd_item<3> item_ct1) {
        count_kernel(N, data, out);
      });
}

// make sure indices is int type
// feat: (b,c,n) indices: (b,n) -> out: (b,c,s), out_indices: (b,n)
// (preprocessed indices)
at::Tensor count_cuda(const at::Tensor idx, const int s) {
  int N = idx.size(0);
  at::Tensor out =
      torch::zeros({s}, at::device(idx.device()).dtype(at::ScalarType::Int));
  count_wrapper(N, idx.data_ptr<int>(), out.data_ptr<int>());
  return out;
}
