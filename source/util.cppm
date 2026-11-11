module;

#include <charconv>
#include <iostream>
#include <optional>
#include <print>
#include <system_error>

export module util;

namespace util {

export template <typename T>
std::optional<T> read_value(std::istream& stream = std::cin) {
  std::string data;
  T result;

  stream >> data;

  auto [ptr, ec] = std::from_chars(data.data(), data.data() + data.size(), result);
  if (ec == std::errc::invalid_argument) {
    if (data.empty()) {
      std::println(stderr, "error: the value required.");
    } else {
      std::println("error: the value {} is not allowed.", data);
    }
    return std::nullopt;
  } else if (ec == std::errc::result_out_of_range) {
    std::println(stderr, "error: the value {} is out of range.", data);
    return std::nullopt;
  }

  return result;
}

}  // namespace util