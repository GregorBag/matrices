#include <gtest/gtest.h>

#include <vector>

import jagged_array_matrix;

TEST(JaggedArrayMatrixTest, IntegerDeterminantTest) {
    struct TestCase {
        std::size_t size;
        std::vector<int> values;
        int det;
    };

    const std::vector<TestCase> test_cases = {
        {3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 0}
    };

    for (const auto& test : test_cases) {
        SCOPED_TRACE("test vector: " + ::testing::PrintToString(test.values));

        matrices::JaggedArrayMatrix<int> matrix{test.size, test.size, test.values.begin(), test.values.end()};
        auto det = matrix.det();
        EXPECT_EQ(det, test.det);
    }
}