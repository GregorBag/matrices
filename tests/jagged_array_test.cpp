#include <gtest/gtest.h>

#include <ranges>
#include <vector>
#include <utility>

import jagged_array;

template <typename T>
struct TestCase {
  std::size_t rows;
  std::size_t cols;
  std::vector<T> values;
};

static const std::vector<TestCase<int>> test_arrays = {
  {3, 4, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
  {5, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};

template <typename T>
array::JaggedArray<T> make_array(const TestCase<T>& test_case) {
  array::JaggedArray<T> result(test_case.rows, test_case.cols);
  for (std::size_t i : std::views::iota(0uz, test_case.rows)) {
    for (std::size_t j : std::views::iota(0uz, test_case.cols)) {
      result[i, j] = test_case.values[i * test_case.cols + j];
    }
  }
  return result;
}

TEST(JaggedArrayTest, InitializerListConstructor) {
  array::JaggedArray<int> array{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(array.rows(), 2);
  EXPECT_EQ(array.cols(), 3);
  for (std::size_t i : std::views::iota(0uz, array.rows())) {
    for (std::size_t j : std::views::iota(0uz, array.cols())) {
      EXPECT_EQ(array[i][j], static_cast<int>(i * array.cols() + j + 1));
    }
  }
}

TEST(JaggedArrayTest, ShapeConstructor) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test_array.values));
    auto array = make_array(test_array);
    for (std::size_t i : std::views::iota(0uz, test_array.rows)) {
      for (std::size_t j : std::views::iota(0uz, test_array.cols)) {
        EXPECT_EQ(test_array.values[i * test_array.cols + j], (array[i, j]));
      }
    }
  }
}

TEST(JaggedArrayTest, CopyConstructor) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test_array.values));
    auto array = make_array(test_array);
    array::JaggedArray<int> copy_array(array);
    EXPECT_EQ(test_array.rows, copy_array.rows());
    EXPECT_EQ(test_array.cols, copy_array.cols());
    for (std::size_t i : std::views::iota(0uz, test_array.rows)) {
      for (std::size_t j : std::views::iota(0uz, test_array.cols)) {
        EXPECT_EQ(test_array.values[i * test_array.cols + j], (copy_array[i, j]));
      }
    }
  }
}

TEST(JaggedArrayTest, MoveConstructor) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test_array.values));
    auto array = make_array(test_array);
    array::JaggedArray<int> move_array(std::move(array));
    EXPECT_EQ(test_array.rows, move_array.rows());
    EXPECT_EQ(test_array.cols, move_array.cols());
    for (std::size_t i : std::views::iota(0uz, test_array.rows)) {
      for (std::size_t j : std::views::iota(0uz, test_array.cols)) {
        EXPECT_EQ(test_array.values[i * test_array.cols + j], (move_array[i, j]));
      }
    }
  }
}

TEST(JaggedArrayTest, CopyAssignment) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test_array.values));
    auto array = make_array(test_array);
    array::JaggedArray<int> copy_array(test_array.rows, test_array.cols, 0);
    copy_array = array;
    EXPECT_EQ(test_array.rows, copy_array.rows());
    EXPECT_EQ(test_array.cols, copy_array.cols());
    for (std::size_t i : std::views::iota(0uz, test_array.rows)) {
      for (std::size_t j : std::views::iota(0uz, test_array.cols)) {
        EXPECT_EQ(test_array.values[i * test_array.cols + j], (copy_array[i, j]));
      }
    }
  }
}

TEST(JaggedArrayTest, MoveAssignment) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test_array.values));
    auto array = make_array(test_array);
    array::JaggedArray<int> move_array(test_array.rows, test_array.cols, 0);
    move_array = std::move(array);
    EXPECT_EQ(test_array.rows, move_array.rows());
    EXPECT_EQ(test_array.cols, move_array.cols());
    for (std::size_t i : std::views::iota(0uz, test_array.rows)) {
      for (std::size_t j : std::views::iota(0uz, test_array.cols)) {
        EXPECT_EQ(test_array.values[i * test_array.cols + j], (move_array[i, j]));
      }
    }
  }
}