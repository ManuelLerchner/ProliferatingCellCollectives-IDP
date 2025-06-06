#pragma once

#include <array>
#include <cstddef>

template <class T, std::size_t SIZE>
[[nodiscard]] constexpr std::array<T, SIZE> add(const std::array<T, SIZE> &a, const std::array<T, SIZE> &b) {
  std::array<T, SIZE> result{};
  for (std::size_t d = 0; d < SIZE; ++d) {
    result[d] = a[d] + b[d];
  }
  return result;
}

/**
 * Adds two arrays, returns the result.
 * @tparam T floating point type
 * @tparam SIZE size of the arrays
 * @param a first summand
 * @param b second summand
 * @return a + b
 */
template <class T, std::size_t SIZE>
constexpr std::array<T, SIZE> operator+(const std::array<T, SIZE> &a, const std::array<T, SIZE> &b) {
  return add(a, b);
}

template <class T, std::size_t SIZE>
[[nodiscard]] constexpr std::array<T, SIZE> negate(const std::array<T, SIZE> &a) {
  std::array<T, SIZE> result{};
  for (std::size_t d = 0; d < SIZE; ++d) {
    result[d] = -a[d];
  }
  return result;
}

/**
 * Negates an array, returns the result.
 * @tparam T floating point type
 * @tparam SIZE size of the array
 * @param a array to negate
 * @return -a
 */
template <class T, std::size_t SIZE>
constexpr std::array<T, SIZE> operator-(const std::array<T, SIZE> &a) {
  return negate(a);
}

// Cross product of two 3D arrays
template <class T>
constexpr std::array<T, 3> cross_product(const std::array<T, 3> &a, const std::array<T, 3> &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
