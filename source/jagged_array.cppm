module;

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <ranges>
#include <utility>

export module jagged_array;

namespace array {

export template <typename T>
class JaggedArray final {
public:
  JaggedArray(std::size_t rows, std::size_t cols, T val = T{}) :
    rows_(rows), cols_(cols), ptr_(rows > 0 ? new T* [rows] {} : nullptr) {
    for (std::size_t i : std::views::iota(0uz, rows)) {
      ptr_[i] = new T[cols]{};
      for (std::size_t j : std::views::iota(0uz, cols)) {
        ptr_[i][j] = val;
      }
    }
  }

  JaggedArray(std::initializer_list<std::initializer_list<T>> values) :
    rows_(values.size()), cols_(0), ptr_(rows_ > 0 ? new T* [rows_] {} : nullptr) {
    for (const auto& row : values) {
      cols_ = std::max(cols_, row.size());
    }

    auto row = values.begin();
    for (std::size_t i : std::views::iota(0uz, rows_)) {
      ptr_[i] = new T[cols_]{};
      std::copy(row->begin(), row->end(), ptr_[i]);
      ++row;
    }
  }

  ~JaggedArray() {
    for (std::size_t i : std::views::iota(0uz, rows_)) {
      delete[] ptr_[i];
    }
    delete[] ptr_;
  }

  JaggedArray(const JaggedArray& rhs) :
    rows_(rhs.rows_), cols_(rhs.cols_), ptr_(rows_ > 0 ? new T*[rows_] : nullptr) {
    for (std::size_t i : std::views::iota(0uz, rows_)) {
      ptr_[i] = new T[cols_];
      std::copy(rhs.ptr_[i], rhs.ptr_[i] + cols_, ptr_[i]);
    }
  }

  JaggedArray(JaggedArray&& rhs) noexcept : rows_(rhs.rows_), cols_(rhs.cols_), ptr_(rhs.ptr_) {
    rhs.rows_ = 0;
    rhs.cols_ = 0;
    rhs.ptr_ = nullptr;
  }

  JaggedArray& operator=(const JaggedArray& rhs) {
    if (&rhs == this) {
      return *this;
    }

    for (std::size_t i : std::views::iota(0uz, rows_)) {
      delete[] ptr_[i];
    }
    delete[] ptr_;

    rows_ = rhs.rows_;
    cols_ = rhs.cols_;

    ptr_ = rows_ > 0 ? new T*[rows_] : nullptr;
    for (std::size_t i : std::views::iota(0uz, rows_)) {
      ptr_[i] = new T[cols_];
      std::copy(rhs.ptr_[i], rhs.ptr_[i] + cols_, ptr_[i]);
    }

    return *this;
  }

  JaggedArray& operator=(JaggedArray&& rhs) noexcept {
    if (&rhs == this) {
      return *this;
    }

    std::swap(rows_, rhs.rows_);
    std::swap(cols_, rhs.cols_);
    std::swap(ptr_, rhs.ptr_);
    return *this;
  }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  T*& operator[](std::size_t i) { return ptr_[i]; }
  const T*& operator[](std::size_t i) const { return ptr_[i]; }
  
  T& operator[](std::size_t i, std::size_t j) { return ptr_[i][j]; }
  const T& operator[](std::size_t i, std::size_t j) const { return ptr_[i][j]; }

private:
  std::size_t rows_, cols_;
  T** ptr_;
};

}  // namespace array