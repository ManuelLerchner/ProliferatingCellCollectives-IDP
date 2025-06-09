#pragma once

#include <array>
#include <cstddef>

namespace utils::ArrayMath {

template <class T, std::size_t SIZE>
[[nodiscard]] constexpr std::array<T, SIZE> add(const std::array<T, SIZE> &a, const std::array<T, SIZE> &b) {
  std::array<T, SIZE> result{};
  for (std::size_t d = 0; d < SIZE; ++d) {
    result[d] = a[d] + b[d];
  }
  return result;
}

template <class T, std::size_t SIZE>
[[nodiscard]] constexpr std::array<T, SIZE> subtract(const std::array<T, SIZE> &a, const std::array<T, SIZE> &b) {
  std::array<T, SIZE> result{};
  for (std::size_t d = 0; d < SIZE; ++d) {
    result[d] = a[d] - b[d];
  }
  return result;
}

template <class T, std::size_t SIZE>
[[nodiscard]] constexpr std::array<T, SIZE> multiply(const std::array<T, SIZE> &a, const std::array<T, SIZE> &b) {
  std::array<T, SIZE> result{};
  for (std::size_t d = 0; d < SIZE; ++d) {
    result[d] = a[d] * b[d];
  }
  return result;
}

template <class T, std::size_t SIZE>
[[nodiscard]] constexpr std::array<T, SIZE> multiply(const std::array<T, SIZE> &a, const T &b) {
  std::array<T, SIZE> result{};
  for (std::size_t d = 0; d < SIZE; ++d) {
    result[d] = a[d] * b;
  }
  return result;
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

/**
 * Subtracts two arrays, returns the result.
 * @tparam T floating point type
 * @tparam SIZE size of the arrays
 * @param a first array
 * @param b second array
 * @return a - b
 */
template <class T, std::size_t SIZE>
constexpr std::array<T, SIZE> operator-(const std::array<T, SIZE> &a, const std::array<T, SIZE> &b) {
  return subtract(a, b);
}

/**
 * Multiplies two arrays, returns the result.
 * @tparam T floating point type
 * @tparam SIZE size of the arrays
 * @param a first array
 * @param b second array
 * @return a * b
 */
template <class T, std::size_t SIZE>
constexpr std::array<T, SIZE> operator*(const std::array<T, SIZE> &a, const std::array<T, SIZE> &b) {
  return multiply(a, b);
}

/**
 * Multiplies an array by a scalar, returns the result.
 * @tparam T floating point type
 * @tparam SIZE size of the array
 * @param a array
 * @param b scalar
 * @return a * b
 */
template <class T, std::size_t SIZE>
constexpr std::array<T, SIZE> operator*(const std::array<T, SIZE> &a, const T &b) {
  return multiply(a, b);
}

/**
 * Multiplies a scalar by an array, returns the result.
 * @tparam T floating point type
 * @tparam SIZE size of the array
 * @param b scalar
 * @param a array
 * @return b * a
 */
template <class T, std::size_t SIZE>
constexpr std::array<T, SIZE> operator*(const T &b, const std::array<T, SIZE> &a) {
  return multiply(a, b);
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

/**
 * Cross product of two 3D arrays
 * @tparam T floating point type
 * @param a first array
 * @param b second array
 * @return a x b
 */
template <class T>
constexpr std::array<T, 3> cross_product(const std::array<T, 3> &a, const std::array<T, 3> &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

template <class T, std::size_t SIZE>
constexpr T dot(const std::array<T, SIZE> &a, const std::array<T, SIZE> &b) {
  T result = 0;
  for (std::size_t d = 0; d < SIZE; ++d) {
    result += a[d] * b[d];
  }
  return result;
}

}  // namespace utils::ArrayMath