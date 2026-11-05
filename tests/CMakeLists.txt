add_executable(
    hwmn_test
    jagged_array_matrix_test.cpp
)

target_link_libraries(
    hwmn_test
    PRIVATE
    matrices
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(hwmn_test)