#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <vector>

import jagged_array_matrix;

namespace {

template <typename T>
matrices::JaggedArrayMatrix<T> make_matrix(std::size_t rows, std::size_t cols,
                                           const std::vector<T>& values) {
  matrices::JaggedArrayMatrix<T> result(rows, cols);
  for (std::size_t i : std::views::iota(0uz, rows)) {
    for (std::size_t j : std::views::iota(0uz, cols)) {
      result[i, j] = values[i * cols + j];
    }
  }
  return result;
}

// Compare elements directly so arithmetic tests do not depend on operator==.
template <typename T>
void expect_matrix(const matrices::JaggedArrayMatrix<T>& matrix, std::size_t rows, std::size_t cols,
                   const std::vector<T>& values) {
  ASSERT_EQ(matrix.rows(), rows);
  ASSERT_EQ(matrix.cols(), cols);
  ASSERT_EQ(values.size(), rows * cols);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      SCOPED_TRACE(::testing::Message() << "row=" << i << ", col=" << j);
      if constexpr (std::is_floating_point_v<T>) {
        EXPECT_DOUBLE_EQ((matrix[i, j]), values[i * cols + j]);
      } else {
        EXPECT_EQ((matrix[i, j]), values[i * cols + j]);
      }
    }
  }
}

struct ArithmeticTestCase {
  std::size_t rows;
  std::size_t cols;
  std::vector<int> lhs;
  std::vector<int> rhs;
  std::vector<int> sum;
  std::vector<int> diff;
};

const std::vector<ArithmeticTestCase> arithmetic_cases = {
  {0, 0, {}, {}, {}, {}},
  {0, 3, {}, {}, {}, {}},
  {3, 0, {}, {}, {}, {}},
  {1, 1, {-7}, {3}, {-4}, {-10}},
  {1, 3, {0, 2, -3}, {4, -2, 1}, {4, 0, -2}, {-4, 4, -4}},
  {3, 1, {0, 2, -3}, {4, -2, 1}, {4, 0, -2}, {-4, 4, -4}},
  {2, 3, {1, -2, 0, 4, 5, -6}, {6, 2, -3, -4, 0, 1}, {7, 0, -3, 0, 5, -5}, {-5, -4, 3, 8, 5, -7}},
  {3, 2, {1, -2, 0, 4, 5, -6}, {6, 2, -3, -4, 0, 1}, {7, 0, -3, 0, 5, -5}, {-5, -4, 3, 8, 5, -7}},
  {2, 2, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
};

}  // namespace

TEST(JaggedArrayMatrixTest, InitializerListConstructor) {
  const matrices::JaggedArrayMatrix<int> matrix{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(matrix.rows(), 2);
  EXPECT_EQ(matrix.cols(), 3);
  for (std::size_t i : std::views::iota(0uz, matrix.rows())) {
    for (std::size_t j : std::views::iota(0uz, matrix.cols())) {
      EXPECT_EQ((matrix[i, j]), static_cast<int>(i * matrix.cols() + j + 1));
    }
  }
}

TEST(JaggedArrayMatrixTest, Negate) {
  struct TestCase {
    std::size_t rows;
    std::size_t cols;
    std::vector<int> values;
    std::vector<int> expected;
  };
  const std::vector<TestCase> cases = {
    {0, 0, {}, {}},
    {0, 3, {}, {}},
    {3, 0, {}, {}},
    {1, 1, {-7}, {7}},
    {1, 3, {0, 2, -3}, {0, -2, 3}},
    {3, 1, {0, 2, -3}, {0, -2, 3}},
    {2, 3, {1, -2, 0, 4, 5, -6}, {-1, 2, 0, -4, -5, 6}},
    {2, 2, {0, 0, 0, 0}, {0, 0, 0, 0}},
    // INT_MIN cannot be negated without signed overflow.
    {1,
     2,
     {std::numeric_limits<int>::max(), -std::numeric_limits<int>::max()},
     {-std::numeric_limits<int>::max(), std::numeric_limits<int>::max()}},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message()
                 << test.rows << 'x' << test.cols << ": " << ::testing::PrintToString(test.values));
    auto matrix = make_matrix(test.rows, test.cols, test.values);
    EXPECT_EQ(&matrix.negate(), &matrix);
    expect_matrix(matrix, test.rows, test.cols, test.expected);
    matrix.negate();
    expect_matrix(matrix, test.rows, test.cols, test.values);
  }
}

TEST(JaggedArrayMatrixTest, Transpose) {
  struct TestCase {
    std::size_t rows;
    std::size_t cols;
    std::vector<int> values;
    std::vector<int> expected;
  };
  const std::vector<TestCase> cases = {
    {0, 0, {}, {}},
    {0, 3, {}, {}},
    {3, 0, {}, {}},
    {1, 1, {-7}, {-7}},
    {1, 3, {1, -2, 3}, {1, -2, 3}},
    {3, 1, {1, -2, 3}, {1, -2, 3}},
    {2, 2, {1, 2, 3, 4}, {1, 3, 2, 4}},
    {2, 3, {1, 2, 3, 4, 5, 6}, {1, 4, 2, 5, 3, 6}},
    {3, 2, {1, 2, 3, 4, 5, 6}, {1, 3, 5, 2, 4, 6}},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message() << test.rows << 'x' << test.cols);
    auto matrix = make_matrix(test.rows, test.cols, test.values);
    EXPECT_EQ(&matrix.transpose(), &matrix);
    expect_matrix(matrix, test.cols, test.rows, test.expected);
    matrix.transpose();
    expect_matrix(matrix, test.rows, test.cols, test.values);
  }
}

TEST(JaggedArrayMatrixTest, AddAssignment) {
  for (const auto& test : arithmetic_cases) {
    SCOPED_TRACE(::testing::Message() << test.rows << 'x' << test.cols);
    auto lhs = make_matrix(test.rows, test.cols, test.lhs);
    const auto rhs = make_matrix(test.rows, test.cols, test.rhs);
    EXPECT_EQ(&(lhs += rhs), &lhs);
    expect_matrix(lhs, test.rows, test.cols, test.sum);
    expect_matrix(rhs, test.rows, test.cols, test.rhs);
  }
}

TEST(JaggedArrayMatrixTest, SubtractAssignment) {
  for (const auto& test : arithmetic_cases) {
    SCOPED_TRACE(::testing::Message() << test.rows << 'x' << test.cols);
    auto lhs = make_matrix(test.rows, test.cols, test.lhs);
    const auto rhs = make_matrix(test.rows, test.cols, test.rhs);
    EXPECT_EQ(&(lhs -= rhs), &lhs);
    expect_matrix(lhs, test.rows, test.cols, test.diff);
    expect_matrix(rhs, test.rows, test.cols, test.rhs);
  }
}

TEST(JaggedArrayMatrixTest, SelfAdditionAndSubtraction) {
  auto matrix = make_matrix<int>(2, 3, {1, -2, 0, 4, 5, -6});
  EXPECT_EQ(&(matrix += matrix), &matrix);
  expect_matrix(matrix, 2, 3, {2, -4, 0, 8, 10, -12});
  EXPECT_EQ(&(matrix -= matrix), &matrix);
  expect_matrix(matrix, 2, 3, {0, 0, 0, 0, 0, 0});
}

TEST(JaggedArrayMatrixTest, Sum) {
  for (const auto& test : arithmetic_cases) {
    SCOPED_TRACE(::testing::Message() << test.rows << 'x' << test.cols);
    const auto lhs = make_matrix(test.rows, test.cols, test.lhs);
    const auto rhs = make_matrix(test.rows, test.cols, test.rhs);
    auto result = matrices::JaggedArrayMatrix<int>::sum(lhs, rhs);
    expect_matrix(result, test.rows, test.cols, test.sum);
    if (test.rows && test.cols) {
      result[0, 0] = 123;
    }
    expect_matrix(lhs, test.rows, test.cols, test.lhs);
    expect_matrix(rhs, test.rows, test.cols, test.rhs);
  }
}

TEST(JaggedArrayMatrixTest, Diff) {
  for (const auto& test : arithmetic_cases) {
    SCOPED_TRACE(::testing::Message() << test.rows << 'x' << test.cols);
    const auto lhs = make_matrix(test.rows, test.cols, test.lhs);
    const auto rhs = make_matrix(test.rows, test.cols, test.rhs);
    auto result = matrices::JaggedArrayMatrix<int>::diff(lhs, rhs);
    expect_matrix(result, test.rows, test.cols, test.diff);
    if (test.rows && test.cols) {
      result[0, 0] = 123;
    }
    expect_matrix(lhs, test.rows, test.cols, test.lhs);
    expect_matrix(rhs, test.rows, test.cols, test.rhs);
  }
}

TEST(JaggedArrayMatrixTest, SumAndDiffWithSameOperand) {
  const auto matrix = make_matrix<int>(2, 3, {1, -2, 0, 4, 5, -6});
  using Matrix = matrices::JaggedArrayMatrix<int>;
  expect_matrix(Matrix::sum(matrix, matrix), 2, 3, {2, -4, 0, 8, 10, -12});
  expect_matrix(Matrix::diff(matrix, matrix), 2, 3, {0, 0, 0, 0, 0, 0});
  expect_matrix(matrix, 2, 3, {1, -2, 0, 4, 5, -6});
}

TEST(JaggedArrayMatrixTest, AdditionAndSubtractionRejectDifferentShapes) {
  using Matrix = matrices::JaggedArrayMatrix<int>;
  // Include equal element counts, a single differing dimension, and empty shapes.
  for (std::size_t rows = 0; rows <= 3; ++rows) {
    for (std::size_t cols = 0; cols <= 3; ++cols) {
      for (std::size_t rhs_rows = 0; rhs_rows <= 3; ++rhs_rows) {
        for (std::size_t rhs_cols = 0; rhs_cols <= 3; ++rhs_cols) {
          if (rows == rhs_rows && cols == rhs_cols) {
            continue;
          }
          SCOPED_TRACE(::testing::Message()
                       << rows << 'x' << cols << " and " << rhs_rows << 'x' << rhs_cols);
          Matrix lhs(rows, cols, 7);
          const Matrix rhs(rhs_rows, rhs_cols, -2);
          EXPECT_THROW(lhs += rhs, std::invalid_argument);
          expect_matrix(lhs, rows, cols, std::vector<int>(rows * cols, 7));
          EXPECT_THROW(lhs -= rhs, std::invalid_argument);
          expect_matrix(lhs, rows, cols, std::vector<int>(rows * cols, 7));
          EXPECT_THROW(Matrix::sum(lhs, rhs), std::invalid_argument);
          EXPECT_THROW(Matrix::diff(lhs, rhs), std::invalid_argument);
          expect_matrix(lhs, rows, cols, std::vector<int>(rows * cols, 7));
          expect_matrix(rhs, rhs_rows, rhs_cols, std::vector<int>(rhs_rows * rhs_cols, -2));
        }
      }
    }
  }
}

TEST(JaggedArrayMatrixTest, Equality) {
  for (const auto& test : arithmetic_cases) {
    SCOPED_TRACE(::testing::Message() << test.rows << 'x' << test.cols);
    const auto lhs = make_matrix(test.rows, test.cols, test.lhs);
    auto rhs = make_matrix(test.rows, test.cols, test.lhs);
    EXPECT_TRUE(lhs == lhs);
    EXPECT_TRUE(lhs == rhs);
    EXPECT_TRUE(rhs == lhs);
    // A difference at any position must be found, including the last element.
    for (std::size_t i = 0; i < test.rows; ++i) {
      for (std::size_t j = 0; j < test.cols; ++j) {
        SCOPED_TRACE(::testing::Message() << "row=" << i << ", col=" << j);
        ++rhs[i, j];
        EXPECT_FALSE(lhs == rhs);
        EXPECT_FALSE(rhs == lhs);
        --rhs[i, j];
      }
    }
  }
}

TEST(JaggedArrayMatrixTest, EqualityRejectsDifferentShapes) {
  for (std::size_t rows = 0; rows <= 3; ++rows) {
    for (std::size_t cols = 0; cols <= 3; ++cols) {
      for (std::size_t rhs_rows = 0; rhs_rows <= 3; ++rhs_rows) {
        for (std::size_t rhs_cols = 0; rhs_cols <= 3; ++rhs_cols) {
          SCOPED_TRACE(::testing::Message()
                       << rows << 'x' << cols << " and " << rhs_rows << 'x' << rhs_cols);
          const matrices::JaggedArrayMatrix<int> lhs(rows, cols, 0);
          const matrices::JaggedArrayMatrix<int> rhs(rhs_rows, rhs_cols, 0);
          EXPECT_EQ(lhs == rhs, rows == rhs_rows && cols == rhs_cols);
        }
      }
    }
  }
}

template <typename T>
class JaggedArrayMatrixFloatingEqualityTest : public ::testing::Test {};

using FloatingTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(JaggedArrayMatrixFloatingEqualityTest, FloatingTypes);

TYPED_TEST(JaggedArrayMatrixFloatingEqualityTest, EqualValuesIncludingNegativeAndZero) {
  using T = TypeParam;
  for (T value : {T{0}, T{1}, T{-1}, T{0.5}, T{-0.5}}) {
    SCOPED_TRACE(::testing::Message() << "value=" << value);
    const matrices::JaggedArrayMatrix<T> lhs(2, 3, value);
    const matrices::JaggedArrayMatrix<T> rhs(2, 3, value);
    EXPECT_TRUE(lhs == lhs);
    EXPECT_TRUE(lhs == rhs);
    EXPECT_TRUE(rhs == lhs);
  }
  const matrices::JaggedArrayMatrix<T> positive_zero(1, 1, T{0});
  const matrices::JaggedArrayMatrix<T> negative_zero(1, 1, -T{0});
  EXPECT_TRUE(positive_zero == negative_zero);
}

TYPED_TEST(JaggedArrayMatrixFloatingEqualityTest, RelativeToleranceIsSymmetric) {
  using T = TypeParam;
  const T epsilon = std::numeric_limits<T>::epsilon();
  for (T value : {T{1}, T{-1}, T{1024}, T{-1024}}) {
    SCOPED_TRACE(::testing::Message() << "value=" << value);
    const matrices::JaggedArrayMatrix<T> lhs(1, 1, value);
    const matrices::JaggedArrayMatrix<T> near(1, 1, std::nextafter(value, T{0}));
    const matrices::JaggedArrayMatrix<T> far(1, 1, value * (T{1} + T{8} * epsilon));
    EXPECT_TRUE(lhs == near);
    EXPECT_TRUE(near == lhs);
    EXPECT_FALSE(lhs == far);
    EXPECT_FALSE(far == lhs);
  }
}

TEST(JaggedArrayMatrixTest, Scale) {
  struct TestCase {
    int factor;
    std::vector<int> expected;
  };
  const std::vector<int> values{1, -2, 0, 4, 5, -6};
  const std::vector<TestCase> cases = {
    {0, {0, 0, 0, 0, 0, 0}},       {1, values},
    {-1, {-1, 2, 0, -4, -5, 6}},   {3, {3, -6, 0, 12, 15, -18}},
    {-2, {-2, 4, 0, -8, -10, 12}},
  };
  using Matrix = matrices::JaggedArrayMatrix<int>;
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message() << "factor=" << test.factor);
    const auto matrix = make_matrix(2, 3, values);
    auto left = Matrix::scale(test.factor, matrix);
    auto right = Matrix::scale(matrix, test.factor);
    expect_matrix(left, 2, 3, test.expected);
    expect_matrix(right, 2, 3, test.expected);
    left[0, 0] = 123;
    right[1, 2] = 456;
    expect_matrix(matrix, 2, 3, values);
  }
  for (const auto& test : arithmetic_cases) {
    const Matrix zero(test.rows, test.cols, 0);
    const std::vector<int> zeros(test.rows * test.cols, 0);
    expect_matrix(Matrix::scale(zero, -3), test.rows, test.cols, zeros);
    expect_matrix(Matrix::scale(-3, zero), test.rows, test.cols, zeros);
  }
}

TEST(JaggedArrayMatrixTest, Product) {
  struct TestCase {
    std::size_t rows;
    std::size_t inner;
    std::size_t cols;
    std::vector<int> lhs;
    std::vector<int> rhs;
    std::vector<int> expected;
  };
  const std::vector<TestCase> cases = {
    {1, 1, 1, {-3}, {4}, {-12}},
    {2, 3, 2, {1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {58, 64, 139, 154}},
    {3,
     2,
     4,
     {1, -2, 0, 3, -1, 4},
     {2, 0, -3, 1, 4, -1, 2, 5},
     {-6, 2, -7, -9, 12, -3, 6, 15, 14, -4, 11, 19}},
    {2, 2, 2, {1, 2, 3, 4}, {0, 1, -1, 2}, {-2, 5, -4, 11}},
    {2, 2, 2, {0, 1, -1, 2}, {1, 2, 3, 4}, {3, 4, 5, 6}},
    {1, 3, 1, {-1, 2, 3}, {4, -1, 6}, {12}},
    {3, 1, 2, {1, -2, 0}, {3, -4}, {3, -4, -6, 8, 0, 0}},
    {2, 2, 3, {0, 0, 0, 0}, {1, 2, 3, 4, 5, 6}, {0, 0, 0, 0, 0, 0}},
    {2, 3, 2, {1, 2, 3, 4, 5, 6}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0}},
    {0, 0, 0, {}, {}, {}},
    {0, 2, 3, {}, {1, 2, 3, 4, 5, 6}, {}},
    {2, 3, 0, {1, 2, 3, 4, 5, 6}, {}, {}},
    {2, 0, 3, {}, {}, {0, 0, 0, 0, 0, 0}},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message()
                 << test.rows << 'x' << test.inner << " times " << test.inner << 'x' << test.cols);
    const auto lhs = make_matrix(test.rows, test.inner, test.lhs);
    const auto rhs = make_matrix(test.inner, test.cols, test.rhs);
    auto result = matrices::JaggedArrayMatrix<int>::product(lhs, rhs);
    expect_matrix(result, test.rows, test.cols, test.expected);
    if (test.rows && test.cols) {
      result[0, 0] = 123;
    }
    expect_matrix(lhs, test.rows, test.inner, test.lhs);
    expect_matrix(rhs, test.inner, test.cols, test.rhs);
  }
}

TEST(JaggedArrayMatrixTest, ProductWithIdentityAndSameOperand) {
  using Matrix = matrices::JaggedArrayMatrix<int>;
  const auto rectangular = make_matrix<int>(2, 3, {1, -2, 0, 4, 5, -6});
  expect_matrix(Matrix::product(Matrix::eye(2), rectangular), 2, 3, {1, -2, 0, 4, 5, -6});
  expect_matrix(Matrix::product(rectangular, Matrix::eye(3)), 2, 3, {1, -2, 0, 4, 5, -6});
  const auto square = make_matrix<int>(2, 2, {1, 2, 3, 4});
  expect_matrix(Matrix::product(square, square), 2, 2, {7, 10, 15, 22});
  expect_matrix(square, 2, 2, {1, 2, 3, 4});
}

TEST(JaggedArrayMatrixTest, ProductRejectsIncompatibleDimensions) {
  using Matrix = matrices::JaggedArrayMatrix<int>;
  for (std::size_t lhs_cols = 0; lhs_cols <= 3; ++lhs_cols) {
    for (std::size_t rhs_rows = 0; rhs_rows <= 3; ++rhs_rows) {
      if (lhs_cols == rhs_rows) {
        continue;
      }
      for (std::size_t outer : {0uz, 2uz}) {
        SCOPED_TRACE(::testing::Message()
                     << outer << 'x' << lhs_cols << " times " << rhs_rows << 'x' << outer);
        const Matrix lhs(outer, lhs_cols, 7);
        const Matrix rhs(rhs_rows, outer, -2);
        EXPECT_THROW(Matrix::product(lhs, rhs), std::invalid_argument);
        expect_matrix(lhs, outer, lhs_cols, std::vector<int>(outer * lhs_cols, 7));
        expect_matrix(rhs, rhs_rows, outer, std::vector<int>(rhs_rows * outer, -2));
      }
    }
  }
}

TEST(JaggedArrayMatrixTest, Trace) {
  struct TestCase {
    std::size_t size;
    std::vector<int> values;
    int expected;
  };
  const std::vector<TestCase> cases = {
    {0, {}, 0},
    {1, {-7}, -7},
    {1, {std::numeric_limits<int>::min()}, std::numeric_limits<int>::min()},
    {1, {std::numeric_limits<int>::max()}, std::numeric_limits<int>::max()},
    {2, {1, 100, -200, 4}, 5},
    {2, {-1, 2, 3, -4}, -5},
    {3, {2, 9, 8, 7, -5, 6, 4, 1, 3}, 0},
    {3, {1, 0, 0, 0, 1, 0, 0, 0, 1}, 3},
    {2, {0, 0, 0, 0}, 0},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test.values));
    const auto matrix = make_matrix(test.size, test.size, test.values);
    const auto trace = matrix.tr();
    ASSERT_TRUE(trace.has_value());
    EXPECT_EQ(*trace, test.expected);
    EXPECT_EQ(matrix.tr(), trace);
    expect_matrix(matrix, test.size, test.size, test.values);
  }
}

TEST(JaggedArrayMatrixTest, RectangularTraceIsAbsent) {
  for (std::size_t rows = 0; rows <= 3; ++rows) {
    for (std::size_t cols = 0; cols <= 3; ++cols) {
      if (rows == cols) {
        continue;
      }
      SCOPED_TRACE(::testing::Message() << rows << 'x' << cols);
      const matrices::JaggedArrayMatrix<int> integers(rows, cols, 1);
      const matrices::JaggedArrayMatrix<double> floating(rows, cols, 0.5);
      EXPECT_FALSE(integers.tr().has_value());
      EXPECT_FALSE(floating.tr().has_value());
    }
  }
}

TEST(JaggedArrayMatrixTest, FractionalArithmetic) {
  using Matrix = matrices::JaggedArrayMatrix<double>;
  const auto lhs = make_matrix<double>(2, 2, {0.5, -1.25, 0.0, 2.5});
  const auto rhs = make_matrix<double>(2, 2, {1.5, 0.25, -2.0, 0.5});
  auto matrix = lhs;
  matrix += rhs;
  expect_matrix(matrix, 2, 2, {2.0, -1.0, -2.0, 3.0});
  matrix = lhs;
  matrix -= rhs;
  expect_matrix(matrix, 2, 2, {-1.0, -1.5, 2.0, 2.0});
  expect_matrix(Matrix::sum(lhs, rhs), 2, 2, {2.0, -1.0, -2.0, 3.0});
  expect_matrix(Matrix::diff(lhs, rhs), 2, 2, {-1.0, -1.5, 2.0, 2.0});
  expect_matrix(Matrix::scale(lhs, -0.5), 2, 2, {-0.25, 0.625, 0.0, -1.25});
  expect_matrix(Matrix::scale(-0.5, lhs), 2, 2, {-0.25, 0.625, 0.0, -1.25});
  expect_matrix(Matrix::product(lhs, rhs), 2, 2, {3.25, -0.5, -5.0, 1.25});
  matrix = lhs;
  expect_matrix(matrix.negate(), 2, 2, {-0.5, 1.25, 0.0, -2.5});
  matrix = lhs;
  expect_matrix(matrix.transpose(), 2, 2, {0.5, 0.0, -1.25, 2.5});
  ASSERT_TRUE(lhs.tr().has_value());
  EXPECT_DOUBLE_EQ(*lhs.tr(), 3.0);
  expect_matrix(lhs, 2, 2, {0.5, -1.25, 0.0, 2.5});
  expect_matrix(rhs, 2, 2, {1.5, 0.25, -2.0, 0.5});
}

TEST(JaggedArrayMatrixTest, WideIntegerArithmetic) {
  using Matrix = matrices::JaggedArrayMatrix<std::int64_t>;
  const auto matrix = make_matrix<std::int64_t>(1, 1, {INT64_C(3000000000)});
  expect_matrix(Matrix::sum(matrix, matrix), 1, 1, {INT64_C(6000000000)});
  expect_matrix(Matrix::scale(matrix, 2), 1, 1, {INT64_C(6000000000)});
  const auto factor = make_matrix<std::int64_t>(1, 1, {2});
  expect_matrix(Matrix::product(matrix, factor), 1, 1, {INT64_C(6000000000)});
  EXPECT_EQ(matrix.tr(), INT64_C(3000000000));
}

// Checks det() on fixed integer matrices of orders 1 through 6:
// exact values and signs, singularity, row swaps, and int boundary values
// without overflowing intermediate calculations.
TEST(JaggedArrayMatrixTest, IntegerDeterminantTest) {
  struct TestCase {
    std::size_t size;
    std::vector<int> values;
    int det;
  };

  const std::vector<TestCase> test_cases = {
    {1, {0}, 0},
    {1, {1}, 1},
    {1, {-7}, -7},
    {1, {42}, 42},
    {1, {std::numeric_limits<int>::min()}, std::numeric_limits<int>::min()},
    {1, {std::numeric_limits<int>::max()}, std::numeric_limits<int>::max()},

    {2, {1, 0, 0, 1}, 1},
    {2, {0, 0, 0, 0}, 0},
    {2, {1, 2, 3, 4}, -2},
    {2, {4, 7, 2, 6}, 10},
    {2, {-2, 3, 4, -5}, -2},
    {2, {1, 2, 2, 4}, 0},
    {2, {0, 1, 1, 0}, -1},
    {2, {0, 2, 0, 3}, 0},
    {2, {0, 0, 2, 3}, 0},
    {2, {2, 3, 0, 0}, 0},
    {2, {2, 0, 3, 0}, 0},
    {2, {0, -2, 3, 4}, 6},
    {2, {-1, 0, 0, -1}, 1},
    // Large inputs with intermediate products that still fit in int.
    {2, {46340, 0, 0, 46340}, 2147395600},
    {2, {std::numeric_limits<int>::min(), 0, 0, 1}, std::numeric_limits<int>::min()},
    {2, {std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max() - 1, 1}, 1},

    {3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 0},
    {3, {1, 0, 0, 0, 1, 0, 0, 0, 1}, 1},
    {3, {2, 3, 4, 0, -3, 5, 0, 0, 4}, -24},
    {3, {2, 0, 0, 3, -3, 0, 4, 5, 4}, -24},
    {3, {1, 2, 3, 1, 2, 3, 4, 5, 6}, 0},
    {3, {1, 2, 3, 0, 0, 0, 4, 5, 6}, 0},
    {3, {0, 1, 0, 0, 0, 1, 1, 0, 0}, 1},
    {3, {1, 1, 0, 1, 1, 1, 0, 1, 1}, -1},
    {3, {6, 1, 1, 4, -2, 5, 2, 8, 7}, -306},
    {3, {0, 0, 0, 0, 0, 0, 0, 0, 0}, 0},
    {3, {1, 1, 1, 1, 1, 1, 1, 1, 1}, 0},
    {3, {-2, 0, 0, 0, -3, 0, 0, 0, -4}, -24},
    // Missing pivots at the first and later elimination steps.
    {3, {0, 1, 2, 0, 3, 4, 0, 5, 6}, 0},
    {3, {1, 0, 2, 0, 0, 3, 0, 0, 4}, 0},
    {3, {1, 2, 3, 2, 4, 7, 3, 6, 10}, 0},
    // A nonzero pivot must be found in the last row.
    {3, {0, 2, 1, 0, 3, 4, 5, 6, 7}, 25},
    {3, {2, 1, 3, 4, 2, 7, 1, 3, 2}, -5},
    // Dense matrices, including a determinant of one.
    {3, {1, 2, 3, 0, 1, 4, 5, 6, 0}, 1},
    {3, {1, 1, 1, 1, 2, 4, 1, 3, 9}, 2},

    {4, {2, 1, 3, 4, 0, -3, 2, 5, 0, 0, 4, 6, 0, 0, 0, 5}, -120},
    {4, {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0}, -1},
    {4, {1, 2, 3, 4, 2, 4, 6, 8, 0, 1, 2, 3, 4, 3, 2, 1}, 0},
    {4, {2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2}, 5},
    {4, {1, 1, 1, 1, 1, 2, 4, 8, 1, 3, 9, 27, 1, 4, 16, 64}, 12},
    {4, {2, 1, 0, 0, 3, 4, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4}, -10},
    {4, {1, 2, 0, 4, 0, 3, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7}, 0},

    {5, {2, -1, 0, 0, 0, -1, 2, -1, 0, 0, 0, -1, 2, -1, 0, 0, 0, -1, 2, -1, 0, 0, 0, -1, 2}, 6},
    {5, {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0}, 1},
    {6,
     {2, -1, 0,  0, 0,  0, -1, 2, -1, 0,  0, 0,  0, -1, 2, -1, 0,  0,
      0, 0,  -1, 2, -1, 0, 0,  0, 0,  -1, 2, -1, 0, 0,  0, 0,  -1, 2},
     7}};

  for (const auto& test : test_cases) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test.values));

    auto matrix = make_matrix(test.size, test.size, test.values);
    auto det = matrix.det();
    EXPECT_EQ(det, test.det);
  }
}

// Enumerates all 625 2x2 matrices over [-2, 2] and all 19683 3x3 matrices
// over [-1, 1]. Checks det() against independent explicit formulas,
// including every singular matrix and sign combination within these ranges.
TEST(JaggedArrayMatrixTest, ExhaustiveSmallIntegerDeterminants) {
  for (std::size_t size : {2uz, 3uz}) {
    const int bound = size == 2 ? 2 : 1;
    const int radix = 2 * bound + 1;
    std::size_t count = 1;
    for (std::size_t i = 0; i < size * size; ++i) {
      count *= radix;
    }
    for (std::size_t code = 0; code < count; ++code) {
      auto digits = code;
      std::vector<int> v(size * size);
      for (auto& value : v) {
        value = static_cast<int>(digits % radix) - bound;
        digits /= radix;
      }
      const int expected = size == 2 ?
                           v[0] * v[3] - v[1] * v[2] :
                           v[0] * (v[4] * v[8] - v[5] * v[7]) - v[1] * (v[3] * v[8] - v[5] * v[6]) +
                           v[2] * (v[3] * v[7] - v[4] * v[6]);
      SCOPED_TRACE("test vector: " + ::testing::PrintToString(v));
      auto matrix = make_matrix(size, size, v);
      EXPECT_EQ(matrix.det(), expected);
    }
  }
}

// Builds all 873 permutation matrices of orders 1 through 6. Checks nonzero
// pivot selection and determinant signs: +1 for even permutations and -1
// for odd permutations. The expected sign is derived from the inversion count.
TEST(JaggedArrayMatrixTest, AllPermutationDeterminantsThroughOrderSix) {
  for (std::size_t size = 1; size <= 6; ++size) {
    std::vector<std::size_t> permutation(size);
    std::iota(permutation.begin(), permutation.end(), 0uz);
    do {
      std::vector<int> values(size * size, 0);
      int inversions = 0;
      for (std::size_t i = 0; i < size; ++i) {
        values[i * size + permutation[i]] = 1;
        for (std::size_t j = i + 1; j < size; ++j) {
          inversions += permutation[i] > permutation[j];
        }
      }
      SCOPED_TRACE("permutation: " + ::testing::PrintToString(permutation));
      auto matrix = make_matrix(size, size, values);
      EXPECT_EQ(matrix.det(), inversions % 2 == 0 ? 1 : -1);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }
}

// Starts with identity matrices of orders 2 through 6 and replaces each row
// or column with zeros or a copy of another row or column. Checks det() = 0
// at every affected position, including the first and last indices.
TEST(JaggedArrayMatrixTest, SingularRowsAndColumnsAtEveryPosition) {
  for (std::size_t size = 2; size <= 6; ++size) {
    for (std::size_t position = 0; position < size; ++position) {
      for (bool column : {false, true}) {
        for (bool duplicate : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << "size=" << size << ", position=" << position << ", column=" << column
                       << ", duplicate=" << duplicate);
          auto matrix = matrices::JaggedArrayMatrix<int>::eye(size);
          const auto other = (position + 1) % size;
          for (std::size_t i = 0; i < size; ++i) {
            if (column) {
              matrix[i, position] = duplicate ? matrix[i, other] : 0;
            } else {
              matrix[position, i] = duplicate ? matrix[other, i] : 0;
            }
          }
          EXPECT_EQ(matrix.det(), 0);
        }
      }
    }
  }
}

// Enumerates all unequal row and column counts from 0 through 4.
// Checks that det() returns an empty optional for nonsquare int and double
// matrices, including shapes 0xN and Nx0.
TEST(JaggedArrayMatrixTest, RectangularDeterminantIsAbsent) {
  for (std::size_t rows = 0; rows <= 4; ++rows) {
    for (std::size_t cols = 0; cols <= 4; ++cols) {
      if (rows == cols) {
        continue;
      }
      SCOPED_TRACE(::testing::Message() << "rows=" << rows << ", cols=" << cols);
      matrices::JaggedArrayMatrix<int> integers{rows, cols, 1};
      matrices::JaggedArrayMatrix<double> floating{rows, cols, 1.0};
      EXPECT_FALSE(integers.det().has_value());
      EXPECT_FALSE(floating.det().has_value());
    }
  }
}

// Creates an integer 0x0 matrix and checks the convention det(0x0) = 1.
// Exposes invalid element access when handling an empty matrix.
// A separate test lets CTest isolate a possible crash.
TEST(JaggedArrayMatrixTest, EmptyIntegerDeterminantIsOne) {
  matrices::JaggedArrayMatrix<int> matrix{0, 0};
  EXPECT_EQ(matrix.det(), 1);
}

// Creates a double 0x0 matrix and checks det(0x0) = 1.0.
// Checks empty matrix handling separately in the floating-point branch.
TEST(JaggedArrayMatrixTest, EmptyFloatingDeterminantIsOne) {
  matrices::JaggedArrayMatrix<double> matrix{0, 0};
  EXPECT_EQ(matrix.det(), 1.0);
}

// Computes a 2x2 int64_t determinant: 100000 * 100000 - 1 = 9999999999.
// Checks a result beyond the 32-bit int range to detect unintended narrowing.
TEST(JaggedArrayMatrixTest, WideIntegerDeterminant) {
  const std::vector<std::int64_t> values{100000, 1, 1, 100000};
  auto matrix = make_matrix<std::int64_t>(2, 2, values);
  EXPECT_EQ(matrix.det(), INT64_C(9999999999));
}

// Checks det() for double: fractional entries, permutations, singularity,
// extreme finite values, different scales, and nearly singular matrices.
// Requires a result with relative tolerance 1e-12 for nonzero determinants
// or absolute tolerance 1e-12 when the expected determinant is zero.
TEST(JaggedArrayMatrixTest, FloatingDeterminantTest) {
  struct TestCase {
    std::size_t size;
    std::vector<double> values;
    double det;
  };
  const std::vector<TestCase> test_cases = {
    {1, {0.0}, 0.0},
    {1, {-0.0}, 0.0},
    {1, {-2.5}, -2.5},
    {1, {1e-100}, 1e-100},
    {1, {1e100}, 1e100},
    {1, {std::numeric_limits<double>::max()}, std::numeric_limits<double>::max()},
    {1, {std::numeric_limits<double>::lowest()}, std::numeric_limits<double>::lowest()},
    {1, {std::numeric_limits<double>::min()}, std::numeric_limits<double>::min()},
    {1, {std::numeric_limits<double>::denorm_min()}, std::numeric_limits<double>::denorm_min()},
    {2, {1.0, 0.0, 0.0, 1.0}, 1.0},
    {2, {0.0, 0.0, 0.0, 0.0}, 0.0},
    {2, {0.5, 0.25, 0.75, 1.0}, 0.3125},
    {2, {0.0, 0.5, 0.25, 1.0}, -0.125},
    {2, {1.0, 2.0, 3.0, 4.0}, -2.0},
    {2, {1.0, 2.0, -3.0, 4.0}, 10.0},
    {2, {1.0, 2.0, 2.0, 4.0}, 0.0},
    {2, {0.0, 2.0, 0.0, 3.0}, 0.0},
    {2, {1e-100, 2e-100, 3e-100, 4e-100}, -2e-200},
    {2, {1e100, 2e100, 3e100, 4e100}, -2e200},
    // Unequal column scales; the determinant is still well within range.
    {2, {1e-100, 0.0, 0.0, 1e100}, 1.0},
    {2, {1e-100, 1e100, 2e-100, 3e100}, 1.0},
    // Exactly representable binary fractions, close to singularity.
    {2, {1.0, 1.0, 1.0, 1.00000095367431640625}, 0.00000095367431640625},
    {2,
     {1.0, 1.0, 1.0, 1.0000000000009094947017729282379150390625},
     0.0000000000009094947017729282379150390625},
    {3, {0, 1, 0, 0, 0, 1, 1, 0, 0}, 1.0},
    {3, {0, 0, 1, 0, 1, 0, 1, 0, 0}, -1.0},
    {3, {0.5, 1, 2, 0, -0.25, 3, 0, 0, 2}, -0.25},
    {3, {6, 1, 1, 4, -2, 5, 2, 8, 7}, -306.0},
    {3, {1, 2, 3, 1, 2, 3, 4, 5, 6}, 0.0},
  };
  for (const auto& test : test_cases) {
    SCOPED_TRACE("test vector: " + ::testing::PrintToString(test.values));
    auto matrix = make_matrix(test.size, test.size, test.values);
    const auto actual = matrix.det();
    ASSERT_TRUE(actual.has_value());
    // Relative tolerance preserves sensitivity for very small determinants.
    const double tolerance = test.det == 0.0 ? 1e-12 : std::abs(test.det) * 1e-12;
    EXPECT_NEAR(*actual, test.det, tolerance);
  }
}

// Calls det() twice on a const matrix that requires row swaps.
// Checks that both results are correct and every original element is preserved:
// decomposition must modify a working copy rather than the input matrix.
TEST(JaggedArrayMatrixTest, DeterminantDoesNotModifyMatrix) {
  const std::vector<int> values{0, 2, 1, 0, 3, 4, 5, 6, 7};
  const auto matrix = make_matrix<int>(3, 3, values);
  EXPECT_EQ(matrix.det(), 25);
  EXPECT_EQ(matrix.det(), 25);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_EQ((matrix[i, j]), values[i * 3 + j]);
    }
  }
}
