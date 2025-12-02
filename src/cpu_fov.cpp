#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <algorithm>

namespace py = pybind11;

namespace {

py::array_t<std::uint8_t> cpu_fov_bool(py::array_t<std::uint8_t> &obstacle,
                                       double x0,
                                       double y0,
                                       double direction_deg,
                                       double vision_length,
                                       double vision_angle_deg,
                                       int num_rays) {
    auto buf_obs = obstacle.request();
    if (buf_obs.ndim != 2) {
        throw std::runtime_error("cpu_fov_bool: obstacle must be 2D");
    }

    const auto height = static_cast<std::int64_t>(buf_obs.shape[0]);
    const auto width = static_cast<std::int64_t>(buf_obs.shape[1]);

    auto *obs_ptr = static_cast<std::uint8_t *>(buf_obs.ptr);

    // Create output mist mask (same shape), initialized to 0
    py::array_t<std::uint8_t> mist(buf_obs.shape);
    auto buf_mist = mist.request();
    auto *mist_ptr = static_cast<std::uint8_t *>(buf_mist.ptr);

    const auto total = height * width;
    std::fill(mist_ptr, mist_ptr + total, static_cast<std::uint8_t>(0));

    if (vision_length <= 0.0 || num_rays <= 0 || width <= 0 || height <= 0) {
        return mist;
    }

    const double half_angle = vision_angle_deg * 0.5;
    const double start_angle = direction_deg - half_angle;
    const double end_angle = direction_deg + half_angle;

    // Precompute angles in radians
    std::vector<double> cos_vals;
    std::vector<double> sin_vals;
    cos_vals.reserve(static_cast<std::size_t>(num_rays));
    sin_vals.reserve(static_cast<std::size_t>(num_rays));

    if (num_rays == 1) {
        const double rad = direction_deg * M_PI / 180.0;
        cos_vals.push_back(std::cos(rad));
        sin_vals.push_back(std::sin(rad));
    } else {
        for (int i = 0; i < num_rays; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(num_rays - 1);
            const double angle_deg = start_angle + t * (end_angle - start_angle);
            const double rad = angle_deg * M_PI / 180.0;
            cos_vals.push_back(std::cos(rad));
            sin_vals.push_back(std::sin(rad));
        }
    }

    const int max_steps = static_cast<int>(std::floor(vision_length));

    for (int r = 0; r < num_rays; ++r) {
        const double cos_a = cos_vals[static_cast<std::size_t>(r)];
        const double sin_a = sin_vals[static_cast<std::size_t>(r)];

        for (int step = 1; step <= max_steps; ++step) {
            const double x = x0 + static_cast<double>(step) * cos_a;
            const double y = y0 + static_cast<double>(step) * sin_a;

            const auto xi = static_cast<std::int64_t>(std::llround(x));
            const auto yi = static_cast<std::int64_t>(std::llround(y));

            if (xi < 0 || yi < 0 || xi >= width || yi >= height) {
                break;
            }

            const auto idx = yi * width + xi;
            mist_ptr[idx] = static_cast<std::uint8_t>(1);

            if (obs_ptr[idx] != 0) {
                break;
            }
        }
    }

    return mist;
}

}  // namespace

PYBIND11_MODULE(cpu_fov, m) {
    m.def(
        "cpu_fov_bool",
        &cpu_fov_bool,
        py::arg("obstacle"),
        py::arg("x0"),
        py::arg("y0"),
        py::arg("direction_deg"),
        py::arg("vision_length"),
        py::arg("vision_angle_deg"),
        py::arg("num_rays") = 72,
        "Compute a simple LiDAR-style FOV mask given obstacle map and agent pose.\n\n"
        "Args:\n"
        "  obstacle: uint8 2D array, 1 indicates blocking cell.\n"
        "  x0, y0: agent position in continuous coordinates.\n"
        "  direction_deg: agent heading in degrees (0 along +x, clockwise).\n"
        "  vision_length: maximum vision radius in pixels.\n"
        "  vision_angle_deg: field of view angle in degrees.\n"
        "  num_rays: number of rays to cast within FOV.\n\n"
        "Returns:\n"
        "  uint8 2D array (same shape as obstacle) where 1 marks visible cells.");
}
