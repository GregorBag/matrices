#include <cstddef>
#include <iostream>
#include <optional>
#include <print>
#include <ranges>
#include <string>
#include <vector>

import jagged_array_matrix;
import util;

int main() {
  auto result = util::read_value<std::size_t>();
  if (!result) {
    return 1;
  }

  auto n = result.value();
  std::vector<double> v;

  std::size_t element_count = 0;
  for (auto _ : std::views::iota(0uz, n * n)) {
    auto value = util::read_value<double>();
    if (!value) {
      std::println(stderr, "error: expected {} arguments for {}x{} matrix, got {}.", n * n, n, n,
                   element_count);
      return 1;
    }
    v.push_back(*value);
    element_count++;
  }

  std::cin >> std::ws;
  if (std::cin.peek() != std::char_traits<char>::eof()) {
    std::println(stderr, "error: too many arguments, first {} were expected.", n * n);
    return 1;
  }

  matrices::JaggedArrayMatrix<double> matrix(n, n, v.begin(), v.end());

  auto d = matrix.det();
  if (!d) {
    std::println(stderr, "error: is not a square matrix");
    return 1;
  }

  std::println("{}", *d);
  return 0;
}
