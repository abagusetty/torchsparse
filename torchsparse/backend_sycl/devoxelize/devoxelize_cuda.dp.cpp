#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <dpct/dpl_utils.hpp>

#include <torch/extension.h>

//#include <THC/THCAtomics.cuh>

// input features (n, c), indices (N, 8), weight (N, 8) -> output features (N,
// c)
template <typename scalar_t>
void devoxelize_forward_kernel(int N, int c,
                                          const int *__restrict__ indices,
                                          const scalar_t *__restrict__ weight,
                                          const scalar_t *__restrict__ feat,
                                          scalar_t *__restrict__ out) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
  int i = index / c;
  int j = index % c;

  if (i < N) {
    const int *indices_ = indices + 8 * i;
    const scalar_t *weight_ = weight + 8 * i;
    const scalar_t *feat_ = feat + j;

    scalar_t cur_feat;
    for (int k = 0; k < 8; k++) {
      cur_feat = 0;
      if (indices_[k] >= 0) cur_feat = feat_[indices_[k] * c];

      out[i * c + j] += weight_[k] * cur_feat;
    }
  }
}

// input weight (N, 8), indices (N, 8), top_grad (N, c) -> bottom grad (n, c)
template <typename scalar_t>
void devoxelize_backward_kernel(
    int N, int n, int c, const int *__restrict__ indices,
    const scalar_t *__restrict__ weight, const scalar_t *__restrict__ top_grad,
    scalar_t *__restrict__ bottom_grad) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
  int i = index / c;
  int j = index % c;

  if (i < N) {
    const int *indices_ = indices + 8 * i;
    const scalar_t *weight_ = weight + 8 * i;

    scalar_t cur_top_grad = top_grad[i * c + j];

#pragma unroll
    for (int k = 0; k < 8; k++) {
      if (indices_[k] >= 0)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &bottom_grad[indices_[k] * c + j], weight_[k] * cur_top_grad);
    }
  }
}

// make sure indices is int type
// feat: (b,c,s) indices: (N, 3) batch_index: (N, ) -> out: (N, c)
at::Tensor devoxelize_forward_cuda(const at::Tensor feat,
                                   const at::Tensor indices,
                                   const at::Tensor weight) {
  int c = feat.size(1);
  int N = indices.size(0);

  at::Tensor out =
      torch::zeros({N, c}, at::device(feat.device()).dtype(feat.dtype()));

  /*
  DPCT1038:0: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat.scalar_type(),
                                      "devoxelize_forward_cuda", ([&] {
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      const int *indices_data_ptr_int_ct2 = indices.data_ptr<int>();
      auto weight_data_ptr_scalar_t_ct3 = weight.data_ptr<scalar_t>();
      auto feat_data_ptr_scalar_t_ct4 = feat.data_ptr<scalar_t>();
      auto out_data_ptr_scalar_t_ct5 = out.data_ptr<scalar_t>();

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, c),
                            sycl::range<3>(1, 1, c)),
          [=](sycl::nd_item<3> item_ct1) {
            devoxelize_forward_kernel<scalar_t>(
                N, c, indices_data_ptr_int_ct2, weight_data_ptr_scalar_t_ct3,
                feat_data_ptr_scalar_t_ct4, out_data_ptr_scalar_t_ct5);
          });
    });
                                      }));

  return out;
}

// top_grad: (N, c), indices: (N, 3), batch_index: (N, ) -> bottom_grad:
// (b,c,s), s=r^3
at::Tensor devoxelize_backward_cuda(const at::Tensor top_grad,
                                    const at::Tensor indices,
                                    const at::Tensor weight, int n) {
  int c = top_grad.size(1);
  int N = top_grad.size(0);
  at::Tensor bottom_grad = torch::zeros(
      {n, c}, at::device(top_grad.device()).dtype(top_grad.dtype()));

  /*
  DPCT1038:2: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(top_grad.scalar_type(),
                                      "devoxelize_backward_cuda", ([&] {
        /*
        DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      const int *indices_data_ptr_int_ct3 = indices.data_ptr<int>();
      auto weight_data_ptr_scalar_t_ct4 = weight.data_ptr<scalar_t>();
      auto top_grad_data_ptr_scalar_t_ct5 = top_grad.data_ptr<scalar_t>();
      auto bottom_grad_data_ptr_scalar_t_ct6 = bottom_grad.data_ptr<scalar_t>();

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, c),
                            sycl::range<3>(1, 1, c)),
          [=](sycl::nd_item<3> item_ct1) {
            devoxelize_backward_kernel<scalar_t>(
                N, n, c, indices_data_ptr_int_ct3, weight_data_ptr_scalar_t_ct4,
                top_grad_data_ptr_scalar_t_ct5,
                bottom_grad_data_ptr_scalar_t_ct6);
          });
    });
                                      }));

  return bottom_grad;
}
