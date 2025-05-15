#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include <torch/extension.h>

/** Reserved value for indicating "empty". */
#define EMPTY_CELL (0)
/** CUDA naive thread block size. */
#define BLOCK_SIZE (256)

__inline__ int8_t atomicCAS(int8_t* address, int8_t compare, int8_t val) {
  int32_t* base_address = (int32_t*)((char*)address - ((size_t)address & 3));
  int32_t int_val = (int32_t)val << (((size_t)address & 3) * 8);
  int32_t int_comp = (int32_t)compare << (((size_t)address & 3) * 8);
  return (int8_t)dpct::atomic_compare_exchange_strong<
      sycl::access::address_space::generic_space>(base_address, int_comp,
                                                  int_val);
}

// TODO: can we do this more efficiently?
__inline__ int16_t atomicCAS(int16_t* address, int16_t compare, int16_t val) {
  int32_t* base_address = (int32_t*)((char*)address - ((size_t)address & 2));
  int32_t int_val = (int32_t)val << (((size_t)address & 2) * 8);
  int32_t int_comp = (int32_t)compare << (((size_t)address & 2) * 8);
  return (int16_t)dpct::atomic_compare_exchange_strong<
      sycl::access::address_space::generic_space>(base_address, int_comp,
                                                  int_val);
}

__inline__ int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val) {
  return (int64_t)dpct::atomic_compare_exchange_strong<
      sycl::access::address_space::generic_space>((unsigned long long *)address,
                                                  (unsigned long long)compare,
                                                  (unsigned long long)val);
}

template <typename dtype=int>
uint64_t hash_func_64b(dtype* data){
  uint64_t hash = 14695981039346656037UL;
  for (int j = 0; j < 4; j++) {
    hash ^= (unsigned int)data[j];
    hash *= 1099511628211UL;
  }
  // hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
  return hash;
}

template <typename key_type>
int hash(key_type key, int _capacity){
  return (uint64_t)key % _capacity;
}

template <typename key_type>
int hash_murmur3(key_type key, int _capacity){
  // use the murmur3 hash function for int32
  int64_t k = (int64_t)key;
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k % _capacity;
}

template <typename key_type, typename val_type>
class GPUHashTable {
 private:
  bool free_pointers;
  const int _capacity;
  const int _divisor;
  key_type* table_keys;
  val_type* table_vals;
  void insert_many_coords(int *coords, const int n);
  void lookup_many_coords(int *coords, val_type *results, 
    const int* kernel_sizes, const int* tensor_strides,
    const int n, const int kernel_volume);
 public:
  GPUHashTable(const int capacity)
      : _capacity(capacity), free_pointers(true), _divisor(128) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    srand(time(NULL));
    table_keys =
        (key_type *)sycl::malloc_device(_capacity * sizeof(key_type), q_ct1);
    q_ct1.memset(table_keys, 0, sizeof(key_type) * _capacity).wait();
    table_vals =
        (val_type *)sycl::malloc_device(_capacity * sizeof(val_type), q_ct1);
    q_ct1.memset(table_vals, 0, sizeof(val_type) * _capacity).wait();
  };
  GPUHashTable(torch::Tensor table_keys, torch::Tensor table_vals)
      : _capacity(table_keys.size(0)), free_pointers(false), table_keys(table_keys.data_ptr<key_type>()),
      table_vals(table_vals.data_ptr<val_type>()), _divisor(128){};
  ~GPUHashTable() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    if(free_pointers){
      dpct::dpct_free(table_keys, q_ct1);
      dpct::dpct_free(table_vals, q_ct1);
    }
  };
  void insert_many(const key_type *keys, const int n);
  void lookup_many(const key_type *keys, val_type *results, const int n);
  void insert_vals(torch::Tensor keys);
  torch::Tensor lookup_vals(torch::Tensor keys);
  void insert_coords(torch::Tensor coords);
  torch::Tensor lookup_coords(at::Tensor coords, at::Tensor kernel_sizes, at::Tensor tensor_strides, int kernel_volume);
  int get_divisor(){return _divisor;}
  int get_capacity(){return _capacity;}
  class device_view{
    private:
      int _capacity;
      key_type* _table_keys;
      val_type* _table_vals;
    public:
      device_view(
        int capacity, key_type* table_keys, val_type* table_vals
      ):_capacity(capacity), _table_keys(table_keys), _table_vals(table_vals){}
      val_type lookup(const key_type key);
      void insert(const key_type key, const val_type val);
  };
  device_view get_device_view(){
    return device_view(_capacity, table_keys, table_vals);
  }
};

using hashtable = GPUHashTable<int64_t, int>;
using hashtable32 = GPUHashTable<int, int>;

// Insert into hashmap
template <typename key_type=int64_t, typename val_type=int>
void insert_kernel(key_type* table_keys, val_type* table_vals, const key_type* keys, int n, int _capacity)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx < n)
    {

        key_type key = keys[idx];
        int value = idx + 1;
        int slot = hash(key, _capacity);
        while (true)
        {
            key_type prev = atomicCAS(&table_keys[slot], EMPTY_CELL, key);
            if (prev == EMPTY_CELL || prev == key)
            {
                table_vals[slot] = value;
                return;
            }

            slot = (slot + 1) % _capacity;
        }
    }
}


template <typename key_type=int64_t, typename val_type=int>
void insert_coords_kernel(key_type* table_keys, val_type* table_vals, int* coords, int n, int _capacity)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx < n)
    {
        key_type key = (key_type)(hash_func_64b(coords + idx * 4));
        int value = idx + 1;
        int slot = hash(key, _capacity);
        while (true)
        {
            key_type prev = atomicCAS(&table_keys[slot], EMPTY_CELL, key);
            if (prev == EMPTY_CELL || prev == key)
            {
                table_vals[slot] = value;
                return;
            }
            slot = (slot + 1) % _capacity;
        }
    }
}


// lookup from hashmap
template <typename key_type=int64_t, typename val_type=int>
void lookup_kernel(key_type* table_keys, val_type* table_vals, const key_type* keys, val_type* vals, int n, int _capacity)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx < n)
    {
        key_type key = keys[idx];
        int slot = hash(key, _capacity);

        while (true)
        {
            key_type cur_key = table_keys[slot];
            if (key == cur_key)
            { 
                vals[idx] = table_vals[slot];
            }
            if (table_keys[slot] == EMPTY_CELL)
            {
                return;
            }
            slot = (slot + 1) % _capacity;
        }
    }
}


template <typename key_type=int64_t, typename val_type=int, bool odd>
void lookup_coords_kernel(
  key_type* table_keys, val_type* table_vals, int* coords, val_type* vals, 
  const int* kernel_sizes, const int* strides, 
  int n, int _capacity, int kernel_volume)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int tidx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2);
    int idx = tidx / kernel_volume;
    int _kernel_idx = tidx % kernel_volume;
    int kernel_idx = _kernel_idx;
    int* in_coords = coords + 4 * idx;
    int coords_out[4];
    coords_out[3] = in_coords[3];
    
    if constexpr (odd) 
    {
      #pragma unroll
      for(int i = 0; i <= 2; i++){
        int cur_offset = _kernel_idx % kernel_sizes[i];
        cur_offset -= (kernel_sizes[i] - 1) / 2;
        coords_out[i] = in_coords[i] * strides[i] + cur_offset;
        _kernel_idx /= kernel_sizes[i];
      }
    }
    else
    {
      #pragma unroll
      for(int i = 2; i >= 0; i--){
        int cur_offset = _kernel_idx % kernel_sizes[i];
        cur_offset -= (kernel_sizes[i] - 1) / 2;
        coords_out[i] = in_coords[i] * strides[i] + cur_offset;
        _kernel_idx /= kernel_sizes[i];
      }
    }
    
    if (idx < n)
    {
        key_type key = (key_type)(hash_func_64b(coords_out));
        int slot = hash(key, _capacity);

        while (true)
        {
            key_type cur_key = table_keys[slot];
            if (key == cur_key)
            { 
                vals[idx * kernel_volume + kernel_idx] = table_vals[slot];
            }
            if (table_keys[slot] == EMPTY_CELL)
            {
                return;
            }
            slot = (slot + 1) % _capacity;
        }
    }
}


template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::insert_many(const key_type *keys, const int n){
  auto exp_props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    auto table_keys_ct0 = table_keys;
    auto table_vals_ct1 = table_vals;
    auto _capacity_ct4 = _capacity;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, (n + BLOCK_SIZE - 1) / BLOCK_SIZE) *
                sycl::range<3>(1, 1, BLOCK_SIZE),
            sycl::range<3>(1, 1, BLOCK_SIZE)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
          insert_kernel<key_type, val_type>(table_keys_ct0, table_vals_ct1,
                                            keys, n, _capacity_ct4);
        });
  });
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::insert_many_coords(int *coords, const int n){
  auto exp_props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    auto table_keys_ct0 = table_keys;
    auto table_vals_ct1 = table_vals;
    auto _capacity_ct4 = _capacity;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, (n + BLOCK_SIZE - 1) / BLOCK_SIZE) *
                sycl::range<3>(1, 1, BLOCK_SIZE),
            sycl::range<3>(1, 1, BLOCK_SIZE)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
          insert_coords_kernel<key_type, val_type>(
              table_keys_ct0, table_vals_ct1, coords, n, _capacity_ct4);
        });
  });
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::insert_vals(at::Tensor keys){
  insert_many(keys.data_ptr<key_type>(), keys.size(0));
}


template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::insert_coords(at::Tensor coords){
  insert_many_coords(coords.data_ptr<int>(), coords.size(0));
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::lookup_many(const key_type *keys, val_type *results, const int n){
  auto exp_props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    auto table_keys_ct0 = table_keys;
    auto table_vals_ct1 = table_vals;
    auto _capacity_ct5 = _capacity;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, (n + BLOCK_SIZE - 1) / BLOCK_SIZE) *
                sycl::range<3>(1, 1, BLOCK_SIZE),
            sycl::range<3>(1, 1, BLOCK_SIZE)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
          lookup_kernel<key_type, val_type>(table_keys_ct0, table_vals_ct1,
                                            keys, results, n, _capacity_ct5);
        });
  });
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::lookup_many_coords(
    int *coords, val_type *results, const int *kernel_sizes, const int *strides,
    const int n, const int kernel_volume) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (kernel_volume % 2)
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
  q_ct1.submit([&](sycl::handler &cgh) {
    auto table_keys_ct0 = table_keys;
    auto table_vals_ct1 = table_vals;
    auto _capacity_ct7 = _capacity;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1,
                           (n * kernel_volume + BLOCK_SIZE - 1) / BLOCK_SIZE) *
                sycl::range<3>(1, 1, BLOCK_SIZE),
            sycl::range<3>(1, 1, BLOCK_SIZE)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
          lookup_coords_kernel<key_type, val_type, true>(
              table_keys_ct0, table_vals_ct1, coords, results, kernel_sizes,
              strides, n, _capacity_ct7, kernel_volume);
        });
  });
  else auto exp_props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  q_ct1.submit([&](sycl::handler &cgh) {
    auto table_keys_ct0 = table_keys;
    auto table_vals_ct1 = table_vals;
    auto _capacity_ct7 = _capacity;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1,
                           (n * kernel_volume + BLOCK_SIZE - 1) / BLOCK_SIZE) *
                sycl::range<3>(1, 1, BLOCK_SIZE),
            sycl::range<3>(1, 1, BLOCK_SIZE)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
          lookup_coords_kernel<key_type, val_type, false>(
              table_keys_ct0, table_vals_ct1, coords, results, kernel_sizes,
              strides, n, _capacity_ct7, kernel_volume);
        });
  });
}

template <typename key_type, typename val_type>
at::Tensor GPUHashTable<key_type, val_type>::lookup_vals(at::Tensor keys){
  auto options =
      torch::TensorOptions().dtype(at::ScalarType::Int).device(keys.device());
  at::Tensor results = torch::zeros({(keys.size(0) + _divisor - 1) / _divisor * _divisor}, options);
  lookup_many(keys.data_ptr<key_type>(), results.data_ptr<val_type>(), keys.size(0));
  return results;
}

template <typename key_type, typename val_type>
at::Tensor GPUHashTable<key_type, val_type>::lookup_coords(at::Tensor coords, at::Tensor kernel_sizes, at::Tensor strides, int kernel_volume){
  auto options =
      torch::TensorOptions().dtype(at::ScalarType::Int).device(coords.device());
  at::Tensor results = torch::zeros({(coords.size(0) + _divisor - 1) / _divisor * _divisor, kernel_volume}, options);
  lookup_many_coords(coords.data_ptr<int>(), results.data_ptr<val_type>(), 
  kernel_sizes.data_ptr<int>(), strides.data_ptr<int>(), coords.size(0), kernel_volume);
  return results;
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::device_view::insert(const key_type key, const val_type val){
  int slot = hash_murmur3(key, _capacity);
  while (true)
  {
    key_type prev = atomicCAS(&_table_keys[slot], EMPTY_CELL, key);
    if (prev == EMPTY_CELL || prev == key)
    {
        _table_vals[slot] = val;
        return;
    }
    slot = (slot + 1) % _capacity;
  }
}

template <typename key_type, typename val_type>
val_type GPUHashTable<key_type, val_type>::device_view::lookup(const key_type key){
  int slot = hash_murmur3(key, _capacity);

  while (true)
  {
    key_type cur_key = _table_keys[slot];
    if (key == cur_key)
    { 
      return _table_vals[slot];
    }
    if (_table_keys[slot] == EMPTY_CELL)
    {
        return EMPTY_CELL;
    }
    slot = (slot + 1) % _capacity;
  }
}