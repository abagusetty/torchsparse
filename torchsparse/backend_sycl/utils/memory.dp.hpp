#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once

template <int bytes>
struct global_load;

template <>
struct global_load<16>
{
  __inline__ global_load(sycl::uint4 &D, void const *ptr, int pred_guard)
  {
    sycl::uint4 &data = *reinterpret_cast<sycl::uint4 *>(&D);
    {
      bool p;
      p = (int)(pred_guard & 1) != 0;
      data.x() = data.x();
      data.y() = data.y();
      data.z() = data.z();
      data.w() = data.w();
      if (p) {
        {data.x(), data.y(), data.z(), data.w()} =
            *((uint32_t *)(uintptr_t)ptr);
      }
    }
  }
};

template <>
struct global_load<8>
{
  __inline__ global_load(sycl::uint4 &D, void const *ptr, int pred_guard)
  {
    sycl::uint2 const *ptr_ldg = reinterpret_cast<sycl::uint2 const *>(ptr);
#pragma unroll
    for (int ldg_idx = 0; ldg_idx < 2; ldg_idx++)
    {
      sycl::uint2 &data = *(reinterpret_cast<sycl::uint2 *>(&D) + ldg_idx);
      {
        bool p;
        p = (int)(pred_guard & (1 << ldg_idx)) != 0;
        data.x() = data.x();
        data.y() = data.y();
        if (p) {
          {data.x(), data.y()} = *((uint32_t *)(uintptr_t)(ptr_ldg + ldg_idx));
        }
      }
    }
  }
};

template <>
struct global_load<4>
{
  __inline__ global_load(sycl::uint4 &D, void const *ptr, int pred_guard)
  {
    unsigned const *ptr_ldg = reinterpret_cast<unsigned const *>(ptr);
#pragma unroll
    for (int ldg_idx = 0; ldg_idx < 4; ldg_idx++)
    {
      unsigned &data = *(reinterpret_cast<unsigned *>(&D) + ldg_idx);
      {
        bool p;
        p = (int)(pred_guard & (1 << ldg_idx)) != 0;
        data = data;
        if (p) {
          data = *(ptr_ldg + ldg_idx);
        }
      }
    }
  }
};

template <>
struct global_load<2>
{
  __inline__ global_load(sycl::uint4 &D, void const *ptr, int pred_guard)
  {
    uint16_t const *ptr_ldg = reinterpret_cast<uint16_t const *>(ptr);
#pragma unroll
    for (int ldg_idx = 0; ldg_idx < 8; ldg_idx++)
    {
      uint16_t &data = *(reinterpret_cast<uint16_t *>(&D) + ldg_idx);
      {
        bool p;
        p = (int)(pred_guard & (1 << ldg_idx)) != 0;
        data = data;
        if (p) {
          data = *(ptr_ldg + ldg_idx);
        }
      }
    }
  }
};

