module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <print>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

export module jagged_array_matrix;

import jagged_array;
import floating_point_comparing;

namespace matrices {

export template <typename T>
struct DecompositionResult;

export template <typename T>
class JaggedArrayMatrix final {
public:
  explicit JaggedArrayMatrix(std::size_t rows, std::size_t cols, T val = T{}) :
    array_(rows, cols, val) {}

  JaggedArrayMatrix(std::initializer_list<std::initializer_list<T>> values) :
    array_(values) {}

  static JaggedArrayMatrix eye(std::size_t n) {
    JaggedArrayMatrix result(n, n, T{0});
    for (std::size_t i : std::views::iota(0uz, n)) {
      result[i, i] = T{1};
    }
    return result;
  }

  T& operator[](std::size_t i, std::size_t j) { return array_[i, j]; }
  const T& operator[](std::size_t i, std::size_t j) const { return array_[i, j]; }

  JaggedArrayMatrix& operator+=(const JaggedArrayMatrix& rhs) & {
    if (!are_same_shaped(rhs)) {
      throw std::invalid_argument("matrices must be same shaped");
    }

    auto rows = array_.rows();
    auto cols = array_.cols();
    for (std::size_t i : std::views::iota(0uz, rows)) {
      for (std::size_t j : std::views::iota(0uz, cols)) {
        (*this)[i, j] += rhs[i, j];
      }
    }

    return *this;
  }

  JaggedArrayMatrix& operator-=(const JaggedArrayMatrix& rhs) & {
    if (!are_same_shaped(rhs)) {
      throw std::invalid_argument("matrices must be same shaped");
    }

    auto rows = array_.rows();
    auto cols = array_.cols();
    for (std::size_t i : std::views::iota(0uz, rows)) {
      for (std::size_t j : std::views::iota(0uz, cols)) {
        (*this)[i, j] -= rhs[i, j];
      }
    }

    return *this;
  }

  JaggedArrayMatrix& operator*=(const T& rhs) & {
    auto rows = array_.rows();
    auto cols = array_.cols();
    for (std::size_t i : std::views::iota(0uz, rows)) {
      for (std::size_t j : std::views::iota(0uz, cols)) {
        (*this)[i, j] *= rhs;
      }
    }

    return *this;
  }

  friend bool operator==(const JaggedArrayMatrix& lhs, const JaggedArrayMatrix& rhs) {
    if (&lhs == &rhs) {
      return true;
    }

    if (!lhs.are_same_shaped(rhs)) {
      return false;
    }

    auto rows = lhs.rows();
    auto cols = lhs.cols();

    for (std::size_t i : std::views::iota(0uz, rows)) {
      for (std::size_t j : std::views::iota(0uz, cols)) {
        if constexpr (std::is_floating_point_v<T>) {
          if (!are_equal(lhs[i, j], rhs[i, j],
                         std::numeric_limits<T>::epsilon() * std::max(lhs[i, j], rhs[i, j]))) {
            return false;
          }
        } else {
          if (lhs[i, j] != rhs[i, j]) {
            return false;
          }
        }
      }
    }

    return true;
  }

  static JaggedArrayMatrix sum(const JaggedArrayMatrix& lhs, const JaggedArrayMatrix& rhs) {
    auto tmp = lhs;
    tmp += rhs;
    return tmp;
  }

  static JaggedArrayMatrix diff(const JaggedArrayMatrix& lhs, const JaggedArrayMatrix& rhs) {
    auto tmp = lhs;
    tmp -= rhs;
    return tmp;
  }

  static JaggedArrayMatrix scale(const T& lhs, const JaggedArrayMatrix& rhs) {
    auto tmp = rhs;
    tmp *= lhs;
    return tmp;
  }

  static JaggedArrayMatrix scale(const JaggedArrayMatrix& lhs, const T& rhs) {
    auto tmp = lhs;
    tmp *= rhs;
    return tmp;
  }

  static JaggedArrayMatrix product(const JaggedArrayMatrix& lhs, const JaggedArrayMatrix& rhs) {
    auto left_rows = lhs.rows();
    auto left_cols = lhs.cols();
    auto right_rows = rhs.rows();
    auto right_cols = rhs.cols();

    if (left_cols != right_rows) {
      throw std::invalid_argument("multiplication is not defined for such matrices");
    }

    JaggedArrayMatrix tmp{left_rows, right_cols, T{0}};
    for (std::size_t i : std::views::iota(0uz, left_rows)) {
      for (std::size_t k : std::views::iota(0uz, left_cols)) {
        for (std::size_t j : std::views::iota(0uz, right_cols)) {
          tmp[i, j] += lhs[i, k] * rhs[k, j];
        }
      }
    }

    return tmp;
  }

  std::size_t rows() const { return array_.rows(); }
  std::size_t cols() const { return array_.cols(); }

  T get_max_elem() const {
    auto rows = array_.rows();
    auto cols = array_.cols();
    T max_elem{0};
    for (std::size_t i : std::views::iota(0uz, rows)) {
      for (std::size_t j : std::views::iota(0uz, cols)) {
        if (max_elem < std::fabs(((*this))[i, j])) {
          max_elem = std::fabs((*this)[i, j]);
        }
      }
    }

    return max_elem;
  }

  bool is_square() const { return array_.rows() == array_.cols(); }
  bool are_same_shaped(const JaggedArrayMatrix& rhs) const {
    return (array_.rows() == rhs.rows() && array_.cols() == rhs.cols());
  }

  void dump(std::ostream& os = std::cout) {
    auto rows = array_.rows();
    auto cols = array_.cols();
    for (std::size_t i : std::views::iota(0uz, rows)) {
      for (std::size_t j : std::views::iota(0uz, cols)) {
        if constexpr (std::is_floating_point_v<T>) {
          std::print(os, "{:8.3f}", (*this)[i, j]);
        } else if constexpr (std::is_integral_v<T>) {
          std::print(os, "{:3}", (*this)[i, j]);
        }
      }
      std::println(os);
    }
  }

  JaggedArrayMatrix& transpose() & {
    auto rows = array_.rows();
    auto cols = array_.cols();
    JaggedArrayMatrix tmp(cols, rows, T{});
    for (std::size_t i : std::views::iota(0uz, rows)) {
      for (std::size_t j : std::views::iota(0uz, cols)) {
        tmp[j, i] = (*this)[i, j];
      }
    }

    std::swap(tmp, *this);
    return *this;
  }

  JaggedArrayMatrix& negate() & {
    auto rows = array_.rows();
    auto cols = array_.cols();
    for (std::size_t i : std::views::iota(0uz, rows)) {
      for (std::size_t j : std::views::iota(0uz, cols)) {
        (*this)[i, j] = -(*this)[i, j];
      }
    }

    return *this;
  }

  JaggedArrayMatrix& swap_rows(size_t i, size_t j) & {
    std::swap(array_[i], array_[j]);
    return *this;
  }

  std::optional<T> tr() const {
    auto rows = array_.rows();
    auto cols = array_.cols();
    if (rows != cols) {
      return std::nullopt;
    }

    T trace = T{0};

    for (std::size_t i : std::views::iota(0uz, rows)) {
      trace += array_[i, i];
    }

    return trace;
  }

  std::optional<DecompositionResult<T>> lup_decompose() const {
    auto rows = array_.rows();
    auto cols = array_.cols();
    if (rows != cols) {
      return std::nullopt;
    }

    std::size_t swaps = 0;
    JaggedArrayMatrix a = *this;

    std::vector<T> column_scale(cols, T{0});
    for (std::size_t i : std::views::iota(0uz, rows)) {
      for (std::size_t j : std::views::iota(0uz, cols)) {
        column_scale[j] = std::max(column_scale[j], std::abs(a[i, j]));
      }
    }

    for (std::size_t k : std::views::iota(0uz, rows)) {
      std::size_t k_prime = k;
      T p = T{0};
      for (std::size_t i : std::views::iota(k, rows)) {
        if (std::abs(a[i, k]) > p) {
          p = std::abs(a[i, k]);
          k_prime = i;
        }
      }

      if (cmp::are_equal(
          p, T{0}, column_scale[k] * static_cast<T>(rows) * std::numeric_limits<T>::epsilon())) {
        return DecompositionResult<T>{std::move(a), swaps, true};
      }

      if (k != k_prime) {
        a.swap_rows(k, k_prime);
        ++swaps;
      }

      for (std::size_t i : std::views::iota(k + 1, rows)) {
        a[i, k] /= a[k, k];
        for (std::size_t j : std::views::iota(k + 1, rows)) {
          a[i, j] -= a[i, k] * a[k, j];
        }
      }
    }

    return DecompositionResult<T>{std::move(a), swaps, false};
  }

  std::optional<DecompositionResult<T>> bareiss_decompose() const {
    auto rows = array_.rows();
    auto cols = array_.cols();
    if (rows != cols) {
      return std::nullopt;
    }

    std::size_t swaps = 0;
    T pivot = T{1};
    JaggedArrayMatrix a = *this;

    if (rows < 2) {
      return DecompositionResult<T>{std::move(a), swaps, false};
    }

    for (std::size_t k : std::views::iota(0uz, rows - 1)) {
      std::size_t r{k};
      while (r < rows && a[r, k] == 0) {
        ++r;
      }

      if (r == rows) {
        return DecompositionResult<T>{std::move(a), swaps, true};
      }

      if (r != k) {
        a.swap_rows(k, r);
        ++swaps;
      }

      for (std::size_t i : std::views::iota(k + 1, rows)) {
        for (std::size_t j : std::views::iota(k + 1, rows)) {
          a[i, j] = (a[i, j] * a[k, k] - a[i, k] * a[k, j]) / pivot;
        }
      }

      pivot = a[k, k];
    }

    return DecompositionResult<T>{std::move(a), swaps, false};
  }

  std::optional<T> det() const {
    if (!is_square()) {
      return std::nullopt;
    }

    if (!rows()) {
      return T{1};
    }

    T det = T{1};

    if constexpr (std::is_floating_point_v<T>) {
      auto lup = lup_decompose();
      if (!lup) {
        return std::nullopt;
      }

      auto& [decomp, swaps, is_singular] = *lup;
      if (is_singular) {
        return T{0};
      }

      auto sign = (swaps % 2) ? T{-1} : T{1};
      auto n = decomp.rows();
      for (std::size_t i : std::views::iota(0uz, n)) {
        det *= decomp[i, i];
      }

      return sign * det;
    } else if constexpr (std::is_integral_v<T>) {
      auto bareiss = bareiss_decompose();
      if (!bareiss) {
        return std::nullopt;
      }

      auto& [decomp, swaps, is_singular] = *bareiss;
      if (is_singular) {
        return T{0};
      }

      auto sign = (swaps % 2) ? T{-1} : T{1};
      auto n = decomp.rows();

      return sign * decomp[n - 1, n - 1];
    }

    return std::nullopt;
  }

private:
  array::JaggedArray<T> array_;
};

export template <typename T>
struct DecompositionResult {
  JaggedArrayMatrix<T> matrix;
  std::size_t swaps;
  bool is_singular;
};

}  // namespace matrices
