module;

#include <cmath>

export module floating_point_comparing;

namespace cmp {

export template <typename FloatT>
bool are_equal(FloatT a, FloatT b, FloatT epsilon) {
  FloatT diff = std::abs(a - b);

  if (diff <= epsilon) {
    return true;
  }
  return false;
}

}  // namespace cmp