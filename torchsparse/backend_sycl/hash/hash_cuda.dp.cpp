#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>
// hashing
// input N*4 int32 tensor output N*1 int64 tensor
void hash_kernel(int N, const int *__restrict__ data,
                            int64_t *__restrict__ out) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);
  if (i < N) {
    data += i * 4;
    uint64_t hash = 14695981039346656037UL;
    for (int j = 0; j < 4; j++) {
      hash ^= (unsigned int)data[j];
      hash *= 1099511628211UL;
    }
    // hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
    out[i] = hash;
  }
}

// kernel hashing: given data D and offset map K, generate D x K
// input N*4 int32 tensor, |K|*3 int32 tensor, output |K|*N int64 tensor
void kernel_hash_kernel(int N, int K, const int *__restrict__ data,
                                   const int *__restrict__ kernel_offset,
                                   int64_t *__restrict__ out,
                                   uint8_t *dpct_local) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto kernel_offset_local = (int *)dpct_local;

  for (int i = 0; i < K * 3; i++) {
    kernel_offset_local[i] = kernel_offset[i];
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);

  int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
  int k = idx % K;
  int i = idx / K;
  int cur_coord[4];
  if (i < N) {
    data += i * 4;
    for (int j = 1; j < 4; j++) {
      cur_coord[j] = data[j] + kernel_offset[k * 3 + j - 1];
    }
    cur_coord[0] = data[0];
    uint64_t hash = 14695981039346656037UL;
    for (int j = 0; j < 4; j++) {
      hash ^= (unsigned int)cur_coord[j];
      hash *= 1099511628211UL;
    }
    // hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
    out[k * N + i] = hash;
  }
}

void kernel_hash_wrapper(int N, int K, const int *data,
                         const int *kernel_offset, int64_t *out) {
  /*
  DPCT1049:513: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto exp_props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    /*
    DPCT1083:567: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(K * 3 * sizeof(int)), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ceil((double)(N * K) / 512)) *
                              sycl::range<3>(1, 1, 512),
                          sycl::range<3>(1, 1, 512)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
          kernel_hash_kernel(
              N, K, data, kernel_offset, out,
              dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
}

void hash_wrapper(int N, const int *data, int64_t *out) {
  /*
  DPCT1049:514: The work-group size passed to the SYCL kernel may exceed the
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
        hash_kernel(N, data, out);
      });
}

at::Tensor hash_cuda(const at::Tensor idx) {
  int N = idx.size(0);
  at::Tensor out =
      torch::zeros({N}, at::device(idx.device()).dtype(at::ScalarType::Long));
  hash_wrapper(N, idx.data_ptr<int>(), out.data_ptr<int64_t>());
  return out;
}

at::Tensor kernel_hash_cuda(const at::Tensor idx,
                            const at::Tensor kernel_offset) {
  int N = idx.size(0);
  int K = kernel_offset.size(0);
  at::Tensor out = torch::zeros(
      {K, N}, at::device(idx.device()).dtype(at::ScalarType::Long));
  kernel_hash_wrapper(N, K, idx.data_ptr<int>(), kernel_offset.data_ptr<int>(),
                      out.data_ptr<int64_t>());
  return out;
}
