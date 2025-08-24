#ifndef PLSCAN_API_ARRAY_H
#define PLSCAN_API_ARRAY_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <memory>

namespace nb = nanobind;

template <typename scalar_t, int N>
using ndarray_ref = nb::ndarray<scalar_t, nb::ndim<N>, nb::numpy, nb::c_contig>;

template <typename scalar_t>
using array_ref = nb::ndarray<scalar_t, nb::ndim<1>, nb::numpy, nb::c_contig>;

template <typename scalar_t>
void pointer_deleter(void *ptr) noexcept {
  delete[] static_cast<scalar_t const *>(ptr);
}

template <typename scalar_t>
array_ref<scalar_t> new_array(size_t const size) {
  auto ptr = std::make_unique<scalar_t[]>(size).release();
  return {ptr, {size}, nb::capsule{ptr, pointer_deleter<scalar_t>}};
}

template <typename scalar_t>
auto new_buffer(size_t const size) {
  auto ptr = std::make_unique<scalar_t[]>(size).release();
  auto capsule = nb::capsule{ptr, pointer_deleter<scalar_t>};
  return std::make_pair(std::span(ptr, size), std::move(capsule));
}

template <typename scalar_t>
array_ref<scalar_t const> to_array(
    std::span<scalar_t> const array, nb::capsule const capsule,
    size_t const new_size
) {
  return {array.data(), {new_size}, capsule};
}

template <typename scalar_t>
std::span<scalar_t> to_view(array_ref<scalar_t> const array) {
  return {array.data(), array.size()};
}

template <typename scalar_t>
std::span<scalar_t> row_view(
    nb::ndarray_view<scalar_t, 2, 'C'> const array, size_t const idx
) {
  return {&array(idx, 0), array.shape(1)};
}

#endif  // PLSCAN_API_ARRAY_H
