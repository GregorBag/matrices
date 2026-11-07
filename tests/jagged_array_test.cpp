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

TEST(JaggedArrayTest, RangeConstructor) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test_array.values));
    array::JaggedArray<int> array(test_array.rows, test_array.cols, test_array.values.begin(),
                                  test_array.values.end());
    for (std::size_t i : std::views::iota(0uz, test_array.rows)) {
      for (std::size_t j : std::views::iota(0uz, test_array.cols)) {
        EXPECT_EQ(test_array.values[i * test_array.cols + j], array[i][j]);
      }
    }
  }
}

TEST(JaggedArrayTest, CopyConstructor) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test_array.values));
    array::JaggedArray<int> array(test_array.rows, test_array.cols, test_array.values.begin(),
                                  test_array.values.end());
    array::JaggedArray<int> copy_array(array);
    EXPECT_EQ(test_array.rows, copy_array.rows());
    EXPECT_EQ(test_array.cols, copy_array.cols());
    for (std::size_t i : std::views::iota(0uz, test_array.rows)) {
      for (std::size_t j : std::views::iota(0uz, test_array.cols)) {
        EXPECT_EQ(test_array.values[i * test_array.cols + j], copy_array[i][j]);
      }
    }
  }
}

TEST(JaggedArrayTest, MoveConstructor) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test_array.values));
    array::JaggedArray<int> array(test_array.rows, test_array.cols, test_array.values.begin(),
                                  test_array.values.end());
    array::JaggedArray<int> move_array(std::move(array));
    EXPECT_EQ(test_array.rows, move_array.rows());
    EXPECT_EQ(test_array.cols, move_array.cols());
    for (std::size_t i : std::views::iota(0uz, test_array.rows)) {
      for (std::size_t j : std::views::iota(0uz, test_array.cols)) {
        EXPECT_EQ(test_array.values[i * test_array.cols + j], move_array[i][j]);
      }
    }
  }
}

TEST(JaggedArrayTest, CopyAssignment) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test_array.values));
    array::JaggedArray<int> array(test_array.rows, test_array.cols, test_array.values.begin(),
                                  test_array.values.end());
    array::JaggedArray<int> copy_array = array;
    EXPECT_EQ(test_array.rows, copy_array.rows());
    EXPECT_EQ(test_array.cols, copy_array.cols());
    for (std::size_t i : std::views::iota(0uz, test_array.rows)) {
      for (std::size_t j : std::views::iota(0uz, test_array.cols)) {
        EXPECT_EQ(test_array.values[i * test_array.cols + j], copy_array[i][j]);
      }
    }
  }
}

TEST(JaggedArrayTest, MoveAssignment) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test_array.values));
    array::JaggedArray<int> array(test_array.rows, test_array.cols, test_array.values.begin(),
                                  test_array.values.end());
    array::JaggedArray<int> move_array = std::move(array);
    EXPECT_EQ(test_array.rows, move_array.rows());
    EXPECT_EQ(test_array.cols, move_array.cols());
    for (std::size_t i : std::views::iota(0uz, test_array.rows)) {
      for (std::size_t j : std::views::iota(0uz, test_array.cols)) {
        EXPECT_EQ(test_array.values[i * test_array.cols + j], move_array[i][j]);
      }
    }
  }
}