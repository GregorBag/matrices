#include <gtest/gtest.h>

#include <initializer_list>
#include <ranges>
#include <utility>
#include <vector>

import jagged_array;

namespace {

template <typename T>
struct TestCase {
  std::size_t rows;
  std::size_t cols;
  std::vector<T> values;
};

static const std::vector<TestCase<int>> test_arrays = {
  {0, 0, {}},
  {0, 4, {}},
  {3, 0, {}},
  {1, 1, {-7}},
  {1, 4, {0, -1, 2, -3}},
  {4, 1, {0, -1, 2, -3}},
  {2, 3, {1, -2, 0, 4, 5, -6}},
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

template <typename T>
void expect_array(const array::JaggedArray<T>& array, const TestCase<T>& expected) {
  ASSERT_EQ(array.rows(), expected.rows);
  ASSERT_EQ(array.cols(), expected.cols);
  ASSERT_EQ(expected.values.size(), expected.rows * expected.cols);
  for (std::size_t i = 0; i < expected.rows; ++i) {
    for (std::size_t j = 0; j < expected.cols; ++j) {
      SCOPED_TRACE(::testing::Message() << "row=" << i << ", col=" << j);
      EXPECT_EQ((array[i, j]), expected.values[i * expected.cols + j]);
    }
  }
}

struct TrackedElement {
  inline static int alive = 0;
  int value = 0;

  TrackedElement() { ++alive; }
  explicit TrackedElement(int v) : value(v) { ++alive; }
  TrackedElement(const TrackedElement& other) : value(other.value) { ++alive; }
  TrackedElement& operator=(const TrackedElement&) = default;
  ~TrackedElement() { --alive; }
};

}  // namespace

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
    SCOPED_TRACE(::testing::Message() << test_array.rows << 'x' << test_array.cols);
    const array::JaggedArray<int> zero(test_array.rows, test_array.cols);
    expect_array(zero, {test_array.rows, test_array.cols,
                        std::vector<int>(test_array.rows * test_array.cols, 0)});
    for (int value : {-7, 0, 42}) {
      SCOPED_TRACE(::testing::Message() << "fill=" << value);
      const array::JaggedArray<int> filled(test_array.rows, test_array.cols, value);
      expect_array(filled, {test_array.rows, test_array.cols,
                            std::vector<int>(test_array.rows * test_array.cols, value)});
    }
  }
}

TEST(JaggedArrayTest, RaggedInitializerListPadsShortRows) {
  const array::JaggedArray<int> array{{1}, {}, {-2, 3, 4}, {5, 6}};
  expect_array(array, {4, 3, {1, 0, 0, 0, 0, 0, -2, 3, 4, 5, 6, 0}});
}

TEST(JaggedArrayTest, EmptyInitializerLists) {
  const array::JaggedArray<int> empty(std::initializer_list<std::initializer_list<int>>{});
  expect_array(empty, {0, 0, {}});
  const array::JaggedArray<int> empty_rows{{}, {}, {}};
  expect_array(empty_rows, {3, 0, {}});
}

TEST(JaggedArrayTest, CopyConstructor) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE(::testing::Message() << test_array.rows << 'x' << test_array.cols << ": "
                                      << ::testing::PrintToString(test_array.values));
    const auto array = make_array(test_array);
    array::JaggedArray<int> copy_array(array);
    expect_array(copy_array, test_array);
    for (std::size_t i = 0; i < test_array.rows; ++i) {
      for (std::size_t j = 0; j < test_array.cols; ++j) {
        ++copy_array[i, j];
      }
    }
    expect_array(array, test_array);
  }
}

TEST(JaggedArrayTest, MoveConstructor) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE(::testing::Message() << test_array.rows << 'x' << test_array.cols);
    auto array = make_array(test_array);
    array::JaggedArray<int> move_array(std::move(array));
    expect_array(move_array, test_array);
    expect_array(array, {0, 0, {}});
    array = array::JaggedArray<int>(1, 2, 42);
    expect_array(array, {1, 2, {42, 42}});
    expect_array(move_array, test_array);
  }
}

TEST(JaggedArrayTest, CopyAssignment) {
  for (const auto& test_array : test_arrays) {
    const auto array = make_array(test_array);
    // Assignment must replace every old shape, including empty destinations.
    for (const auto& old : test_arrays) {
      SCOPED_TRACE(::testing::Message() << old.rows << 'x' << old.cols << " <- " << test_array.rows
                                        << 'x' << test_array.cols);
      auto copy_array = make_array(old);
      EXPECT_EQ(&(copy_array = array), &copy_array);
      expect_array(copy_array, test_array);
      for (std::size_t i = 0; i < test_array.rows; ++i) {
        for (std::size_t j = 0; j < test_array.cols; ++j) {
          ++copy_array[i, j];
        }
      }
      expect_array(array, test_array);
    }
  }
}

TEST(JaggedArrayTest, MoveAssignment) {
  for (const auto& test_array : test_arrays) {
    for (const auto& old : test_arrays) {
      SCOPED_TRACE(::testing::Message() << old.rows << 'x' << old.cols << " <- " << test_array.rows
                                        << 'x' << test_array.cols);
      auto array = make_array(test_array);
      auto move_array = make_array(old);
      EXPECT_EQ(&(move_array = std::move(array)), &move_array);
      expect_array(move_array, test_array);
      // Move assignment swaps storage; the source need not become empty.
      // It must remain assignable without affecting the destination.
      array = array::JaggedArray<int>(1, 2, 42);
      expect_array(array, {1, 2, {42, 42}});
      expect_array(move_array, test_array);
    }
  }
}

TEST(JaggedArrayTest, SelfCopyAndMoveAssignment) {
  for (const auto& test_array : test_arrays) {
    SCOPED_TRACE(::testing::Message() << test_array.rows << 'x' << test_array.cols);
    auto array = make_array(test_array);
    auto& alias = array;
    EXPECT_EQ(&(array = alias), &array);
    expect_array(array, test_array);
    EXPECT_EQ(&(array = std::move(alias)), &array);
    expect_array(array, test_array);
  }
}

TEST(JaggedArrayTest, CopiesOutliveTheirSource) {
  const TestCase<int> test{2, 3, {1, -2, 0, 4, 5, -6}};
  array::JaggedArray<int> assigned(0, 0);
  const auto constructed = [&] {
    auto source = make_array(test);
    array::JaggedArray<int> copy(source);
    assigned = source;
    source[0, 0] = 999;
    expect_array(copy, test);
    expect_array(assigned, test);
    return copy;
  }();
  expect_array(constructed, test);
  expect_array(assigned, test);
}

TEST(JaggedArrayTest, MovedArraysOutliveTheirSource) {
  const TestCase<int> test{2, 3, {1, -2, 0, 4, 5, -6}};
  array::JaggedArray<int> assigned(3, 1, 99);
  {
    auto source = make_array(test);
    assigned = std::move(source);
  }
  expect_array(assigned, test);
  const auto constructed = [&] {
    auto source = make_array(test);
    array::JaggedArray<int> moved(std::move(source));
    return moved;
  }();
  expect_array(constructed, test);
}

TEST(JaggedArrayTest, DestructorReleasesCopiedAndMovedElements) {
  ASSERT_EQ(TrackedElement::alive, 0);
  {
    const TrackedElement fill(7);
    {
      array::JaggedArray<TrackedElement> source(2, 3, fill);
      EXPECT_EQ(TrackedElement::alive, 7);
      {
        array::JaggedArray<TrackedElement> copy(source);
        EXPECT_EQ(TrackedElement::alive, 13);
        array::JaggedArray<TrackedElement> moved(std::move(copy));
        EXPECT_EQ(TrackedElement::alive, 13);
        moved[1, 2].value = 42;
        EXPECT_EQ((source[1, 2].value), 7);
      }
      EXPECT_EQ(TrackedElement::alive, 7);
    }
    EXPECT_EQ(TrackedElement::alive, 1);
  }
  EXPECT_EQ(TrackedElement::alive, 0);
}

TEST(JaggedArrayTest, AssignmentsReleaseReplacedElements) {
  ASSERT_EQ(TrackedElement::alive, 0);
  {
    const TrackedElement fill(7);
    {
      array::JaggedArray<TrackedElement> source(2, 3, fill);
      array::JaggedArray<TrackedElement> destination(1, 2, fill);
      EXPECT_EQ(TrackedElement::alive, 9);
      destination = source;
      EXPECT_EQ(TrackedElement::alive, 13);
      destination = array::JaggedArray<TrackedElement>(0, 0);
      EXPECT_EQ(TrackedElement::alive, 7);
    }
    EXPECT_EQ(TrackedElement::alive, 1);
    {
      array::JaggedArray<TrackedElement> destination(1, 2, fill);
      {
        array::JaggedArray<TrackedElement> source(2, 3, fill);
        destination = std::move(source);
      }
      ASSERT_EQ(destination.rows(), 2);
      ASSERT_EQ(destination.cols(), 3);
      EXPECT_EQ((destination[1, 2].value), 7);
      EXPECT_EQ(TrackedElement::alive, 7);
    }
    EXPECT_EQ(TrackedElement::alive, 1);
  }
  EXPECT_EQ(TrackedElement::alive, 0);
}
