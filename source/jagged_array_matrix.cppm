module;

#include <cmath>
#include <cstddef>
#include <iostream>
#include <optional>
#include <print>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

export module jagged_array_matrix;

import jagged_array;
import floating_point_comparing;

namespace matrices {

export template <typename T>
class JaggedArrayMatrix final {
public:
  explicit JaggedArrayMatrix(std::size_t rows, std::size_t cols, T val = T{}) :
    array_(rows, cols, val) {}

  template <typename It>
  explicit JaggedArrayMatrix(std::size_t rows, std::size_t cols, It start, It finish) :
    array_(rows, cols, start, finish) {}

  static JaggedArrayMatrix eye(std::size_t n) {
    std::vector<T> v(n * n, T{0});
    for (std::size_t i : std::views::iota(0uz, n)) {
      v[i * n + i] = T{1};
    }
    return JaggedArrayMatrix(n, n, v.begin(), v.end());
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

  std::size_t rows() const { return array_.rows(); }
  std::size_t cols() const { return array_.cols(); }

  T get_max_elem() const {
    auto rows = array_.rows();
    auto cols = array_.cols();
    T max_elem = std::fabs(array_[0, 0]);
    for (std::size_t i : std::views::iota(0uz, rows)) {
      for (std::size_t j : std::views::iota(0uz, cols)) {
        if (max_elem > std::fabs((array_[i, j]))) {
          max_elem = std::fabs(array_[i, j]);
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
        std::print(os, "{:8.3f}", array_[i, j]);
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

  std::optional<T> trace() const {
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

  std::optional<T> determinant() {
    auto rows = array_.rows();
    auto cols = array_.cols();
    if (rows != cols) {
      return std::nullopt;
    }

    if (!rows) {
      return T{1};
    }

    T sign = T{1};
    JaggedArrayMatrix a = *this;

    for (std::size_t k : std::views::iota(0uz, rows)) {
      std::size_t k_prime = k;
      T p = T{0};
      for (std::size_t i : std::views::iota(k, rows)) {
        if (std::abs(a[i, k]) > p) {
          p = a[i, k];
          k_prime = i;
        }
      }

      if (cmp::are_equal(p, T{0}, get_max_elem(), rows)) {
        return T{0};
      }

      if (k != k_prime) {
        for (std::size_t i : std::views::iota(0uz, rows)) {
          std::swap(a[k, i], a[k_prime, i]);
        }
        sign = -sign;
      }

      for (std::size_t i : std::views::iota(k + 1, rows)) {
        a[i, k] = a[i, k] / a[k, k];
        for (std::size_t j : std::views::iota(k + 1, rows)) {
          a[i, j] -= a[i, k] * a[k, j];
        }
      }
    }

    T d = sign;
    for (std::size_t k : std::views::iota(0uz, rows)) {
      d *= a[k, k];
    }

    return d;
  }

private:
  array::JaggedArray<T> array_;
};

}  // namespace matrices
