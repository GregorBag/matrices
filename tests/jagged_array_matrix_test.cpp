#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <vector>

import jagged_array_matrix;

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
        {2, {std::numeric_limits<int>::max(), 1,
             std::numeric_limits<int>::max() - 1, 1}, 1},

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

        {4, {2, 1, 3, 4,
             0, -3, 2, 5,
             0, 0, 4, 6,
             0, 0, 0, 5}, -120},
        {4, {0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1,
             1, 0, 0, 0}, -1},
        {4, {1, 2, 3, 4,
             2, 4, 6, 8,
             0, 1, 2, 3,
             4, 3, 2, 1}, 0},
        {4, {2, -1, 0, 0,
             -1, 2, -1, 0,
             0, -1, 2, -1,
             0, 0, -1, 2}, 5},
        {4, {1, 1, 1, 1,
             1, 2, 4, 8,
             1, 3, 9, 27,
             1, 4, 16, 64}, 12},
        {4, {2, 1, 0, 0,
             3, 4, 0, 0,
             0, 0, 1, 2,
             0, 0, 3, 4}, -10},
        {4, {1, 2, 0, 4,
             0, 3, 0, 5,
             0, 0, 0, 6,
             0, 0, 0, 7}, 0},

        {5, {2, -1, 0, 0, 0,
             -1, 2, -1, 0, 0,
             0, -1, 2, -1, 0,
             0, 0, -1, 2, -1,
             0, 0, 0, -1, 2}, 6},
        {5, {0, 1, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 1,
             1, 0, 0, 0, 0}, 1},
        {6, {2, -1, 0, 0, 0, 0,
             -1, 2, -1, 0, 0, 0,
             0, -1, 2, -1, 0, 0,
             0, 0, -1, 2, -1, 0,
             0, 0, 0, -1, 2, -1,
             0, 0, 0, 0, -1, 2}, 7}
    };

    for (const auto& test : test_cases) {
        SCOPED_TRACE("test vector: " + ::testing::PrintToString(test.values));

        matrices::JaggedArrayMatrix<int> matrix{test.size, test.size, test.values.begin(), test.values.end()};
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
            const int expected = size == 2
                ? v[0] * v[3] - v[1] * v[2]
                : v[0] * (v[4] * v[8] - v[5] * v[7])
                    - v[1] * (v[3] * v[8] - v[5] * v[6])
                    + v[2] * (v[3] * v[7] - v[4] * v[6]);
            SCOPED_TRACE("test vector: " + ::testing::PrintToString(v));
            matrices::JaggedArrayMatrix<int> matrix{size, size, v.begin(), v.end()};
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
            matrices::JaggedArrayMatrix<int> matrix{size, size, values.begin(), values.end()};
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
                    SCOPED_TRACE(::testing::Message() << "size=" << size << ", position="
                        << position << ", column=" << column << ", duplicate=" << duplicate);
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
    matrices::JaggedArrayMatrix<std::int64_t> matrix{2, 2, values.begin(), values.end()};
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
        {2, {1.0, 1.0, 1.0, 1.0000000000009094947017729282379150390625},
             0.0000000000009094947017729282379150390625},
        {3, {0, 1, 0, 0, 0, 1, 1, 0, 0}, 1.0},
        {3, {0, 0, 1, 0, 1, 0, 1, 0, 0}, -1.0},
        {3, {0.5, 1, 2, 0, -0.25, 3, 0, 0, 2}, -0.25},
        {3, {6, 1, 1, 4, -2, 5, 2, 8, 7}, -306.0},
        {3, {1, 2, 3, 1, 2, 3, 4, 5, 6}, 0.0},
    };
    for (const auto& test : test_cases) {
        SCOPED_TRACE("test vector: " + ::testing::PrintToString(test.values));
        matrices::JaggedArrayMatrix<double> matrix{test.size, test.size,
            test.values.begin(), test.values.end()};
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
    const matrices::JaggedArrayMatrix<int> matrix{3, 3, values.begin(), values.end()};
    EXPECT_EQ(matrix.det(), 25);
    EXPECT_EQ(matrix.det(), 25);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_EQ((matrix[i, j]), values[i * 3 + j]);
        }
    }
}
