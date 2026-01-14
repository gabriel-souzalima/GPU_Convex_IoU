// python bindings file
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <fstream>
#include <iostream>

struct EllipseData
{
    double x, y, w, h, angle_rad;
};

// external CUDA functions
extern "C" void compute_iou_matrix_cuda(const std::vector<EllipseData> &ellipses,
                                        std::vector<float> &h_matrix,
                                        int num_points);

extern "C" void compute_iou_rectangular_cuda(const std::vector<EllipseData> &rows,
                                             const std::vector<EllipseData> &cols,
                                             std::vector<float> &h_matrix,
                                             int num_points);

extern "C" void compute_iou_batched_rectangular_cuda(
    const std::vector<EllipseData> &all_dets,
    const std::vector<EllipseData> &all_gts,
    const std::vector<int> &pair_info,
    std::vector<float> &h_results,
    int total_output_size,
    int num_points);

namespace py = pybind11;

// helper functions

static std::vector<EllipseData> parse_ellipses_or_throw(const std::vector<std::vector<double>> &raw)
{
    std::vector<EllipseData> out;
    out.reserve(raw.size());
    for (const auto &e : raw)
    {
        if (e.size() < 5)
            throw std::runtime_error("Ellipse data must have 5 elements: x, y, w, h, angle_rad");
        out.push_back({e[0], e[1], e[2], e[3], e[4]});
    }
    return out;
}

static std::vector<EllipseData> parse_ellipses_numpy_or_throw(const py::array &arr_any)
{
    py::array_t<double, py::array::c_style | py::array::forcecast> arr(arr_any);
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] < 5)
        throw std::runtime_error("Expected a 2D array with shape (N,5) (or more columns)");

    const auto n = static_cast<size_t>(buf.shape[0]);
    const auto stride0 = static_cast<ssize_t>(buf.strides[0] / sizeof(double));
    const auto stride1 = static_cast<ssize_t>(buf.strides[1] / sizeof(double));
    const double *ptr = static_cast<const double *>(buf.ptr);

    std::vector<EllipseData> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        const double *row = ptr + static_cast<ssize_t>(i) * stride0;
        out.push_back({row[0 * stride1], row[1 * stride1], row[2 * stride1], row[3 * stride1], row[4 * stride1]});
    }
    return out;
}

// NxN matrix functions
std::vector<float> calculate_iou_matrix(const std::vector<std::vector<double>> &raw_ellipses, int num_points)
{
    std::vector<EllipseData> cpp_ellipses = parse_ellipses_or_throw(raw_ellipses);
    std::vector<float> matrix;
    compute_iou_matrix_cuda(cpp_ellipses, matrix, num_points);
    return matrix;
}

py::array_t<float> calculate_iou_matrix_numpy(const std::vector<std::vector<double>> &raw_ellipses, int num_points)
{
    std::vector<EllipseData> cpp_ellipses = parse_ellipses_or_throw(raw_ellipses);
    const size_t n = cpp_ellipses.size();

    auto *matrix = new std::vector<float>();
    compute_iou_matrix_cuda(cpp_ellipses, *matrix, num_points);
    if (matrix->size() != n * n)
    {
        delete matrix;
        throw std::runtime_error("Unexpected matrix output size");
    }

    py::capsule base(matrix, [](void *v)
                     { delete reinterpret_cast<std::vector<float> *>(v); });
    return py::array_t<float>(
        {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)},
        {static_cast<py::ssize_t>(n * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))},
        matrix->data(),
        base);
}

py::array_t<float> calculate_iou_matrix_numpy_from_numpy(const py::array &ellipses, int num_points)
{
    std::vector<EllipseData> cpp_ellipses = parse_ellipses_numpy_or_throw(ellipses);
    const size_t n = cpp_ellipses.size();

    auto *matrix = new std::vector<float>();
    compute_iou_matrix_cuda(cpp_ellipses, *matrix, num_points);
    if (matrix->size() != n * n)
    {
        delete matrix;
        throw std::runtime_error("Unexpected matrix output size");
    }

    py::capsule base(matrix, [](void *v)
                     { delete reinterpret_cast<std::vector<float> *>(v); });
    return py::array_t<float>(
        {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)},
        {static_cast<py::ssize_t>(n * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))},
        matrix->data(),
        base);
}

// rectangular matrix functions

std::vector<float> calculate_iou_rectangular(const std::vector<std::vector<double>> &raw_rows,
                                             const std::vector<std::vector<double>> &raw_cols,
                                             int num_points)
{
    std::vector<EllipseData> rows = parse_ellipses_or_throw(raw_rows);
    std::vector<EllipseData> cols = parse_ellipses_or_throw(raw_cols);

    std::vector<float> matrix;
    compute_iou_rectangular_cuda(rows, cols, matrix, num_points);
    return matrix;
}

py::array_t<float> calculate_iou_rectangular_numpy(const std::vector<std::vector<double>> &raw_rows,
                                                   const std::vector<std::vector<double>> &raw_cols,
                                                   int num_points)
{
    std::vector<EllipseData> rows = parse_ellipses_or_throw(raw_rows);
    std::vector<EllipseData> cols = parse_ellipses_or_throw(raw_cols);
    const size_t n_rows = rows.size();
    const size_t n_cols = cols.size();

    auto *matrix = new std::vector<float>();
    compute_iou_rectangular_cuda(rows, cols, *matrix, num_points);
    if (matrix->size() != n_rows * n_cols)
    {
        delete matrix;
        throw std::runtime_error("Unexpected rectangular output size");
    }

    py::capsule base(matrix, [](void *v)
                     { delete reinterpret_cast<std::vector<float> *>(v); });
    return py::array_t<float>(
        {static_cast<py::ssize_t>(n_rows), static_cast<py::ssize_t>(n_cols)},
        {static_cast<py::ssize_t>(n_cols * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))},
        matrix->data(),
        base);
}

py::array_t<float> calculate_iou_rectangular_numpy_from_numpy(const py::array &rows, const py::array &cols, int num_points)
{
    std::vector<EllipseData> r = parse_ellipses_numpy_or_throw(rows);
    std::vector<EllipseData> c = parse_ellipses_numpy_or_throw(cols);
    const size_t n_rows = r.size();
    const size_t n_cols = c.size();

    auto *matrix = new std::vector<float>();
    compute_iou_rectangular_cuda(r, c, *matrix, num_points);
    if (matrix->size() != n_rows * n_cols)
    {
        delete matrix;
        throw std::runtime_error("Unexpected rectangular output size");
    }

    py::capsule base(matrix, [](void *v)
                     { delete reinterpret_cast<std::vector<float> *>(v); });
    return py::array_t<float>(
        {static_cast<py::ssize_t>(n_rows), static_cast<py::ssize_t>(n_cols)},
        {static_cast<py::ssize_t>(n_cols * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))},
        matrix->data(),
        base);
}

// batched rectangular

py::tuple calculate_iou_batched_rectangular(
    const py::array &all_dets,      // (N_total_dets, 5)
    const py::array &all_gts,       // (M_total_gts, 5)
    const py::array &pair_info_arr, // (num_pairs, 5): [det_off, gt_off, out_off, n_det, n_gt]
    int num_points)
{
    // Parse all detections
    py::array_t<double, py::array::c_style | py::array::forcecast> dets_arr(all_dets);
    auto dets_buf = dets_arr.request();
    if (dets_buf.ndim != 2 || dets_buf.shape[1] < 5)
        throw std::runtime_error("all_dets must be (N, 5)");

    const size_t n_dets = dets_buf.shape[0];
    const double *dets_ptr = static_cast<const double *>(dets_buf.ptr);
    std::vector<EllipseData> cpp_dets;
    cpp_dets.reserve(n_dets);
    for (size_t i = 0; i < n_dets; ++i)
    {
        const double *row = dets_ptr + i * 5;
        cpp_dets.push_back({row[0], row[1], row[2], row[3], row[4]});
    }

    // parse all ground truths
    py::array_t<double, py::array::c_style | py::array::forcecast> gts_arr(all_gts);
    auto gts_buf = gts_arr.request();
    if (gts_buf.ndim != 2 || gts_buf.shape[1] < 5)
        throw std::runtime_error("all_gts must be (M, 5)");

    const size_t n_gts = gts_buf.shape[0];
    const double *gts_ptr = static_cast<const double *>(gts_buf.ptr);
    std::vector<EllipseData> cpp_gts;
    cpp_gts.reserve(n_gts);
    for (size_t i = 0; i < n_gts; ++i)
    {
        const double *row = gts_ptr + i * 5;
        cpp_gts.push_back({row[0], row[1], row[2], row[3], row[4]});
    }

    // parse pair info
    py::array_t<int, py::array::c_style | py::array::forcecast> info_arr(pair_info_arr);
    auto info_buf = info_arr.request();
    if (info_buf.ndim != 2 || info_buf.shape[1] != 5)
        throw std::runtime_error("pair_info must be (num_pairs, 5)");

    const size_t num_pairs = info_buf.shape[0];
    const int *info_ptr = static_cast<const int *>(info_buf.ptr);
    std::vector<int> pair_info(info_ptr, info_ptr + num_pairs * 5);

    // calculate total output size
    int total_output_size = 0;
    for (size_t p = 0; p < num_pairs; ++p)
    {
        total_output_size += pair_info[p * 5 + 3] * pair_info[p * 5 + 4];
    }

    // call CUDA
    auto *results = new std::vector<float>();
    compute_iou_batched_rectangular_cuda(cpp_dets, cpp_gts, pair_info, *results, total_output_size, num_points);

    // return flat array
    py::capsule base(results, [](void *v)
                     { delete reinterpret_cast<std::vector<float> *>(v); });
    py::array_t<float> result_arr(
        {static_cast<py::ssize_t>(total_output_size)},
        {static_cast<py::ssize_t>(sizeof(float))},
        results->data(),
        base);

    return py::make_tuple(result_arr, total_output_size);
}

// binary file interface
std::vector<float> calculate_iou_matrix_from_file(std::string filename, int num_points)
{
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    int32_t count;
    f.read(reinterpret_cast<char *>(&count), sizeof(int32_t));

    if (count <= 0)
    {
        throw std::runtime_error("Invalid dataset count in file.");
    }

    std::vector<EllipseData> cpp_ellipses(count);
    f.read(reinterpret_cast<char *>(cpp_ellipses.data()), count * sizeof(EllipseData));

    if (!f)
    {
        throw std::runtime_error("Error reading data from file.");
    }

    std::cout << "[C++] Loaded " << count << " ellipses from binary file." << std::endl;

    std::vector<float> matrix;
    compute_iou_matrix_cuda(cpp_ellipses, matrix, num_points);
    return matrix;
}

PYBIND11_MODULE(convexiou_gpu, m)
{
    m.doc() = "GPU-accelerated Convex Polygon IoU Calculator for Ellipse Approximation";

    // NxN Matrix
    m.def("calculate_iou_matrix", &calculate_iou_matrix,
          "Compute NxN IoU Matrix on GPU from Python list",
          py::arg("ellipses"), py::arg("num_points") = 16);

    m.def("calculate_iou_matrix_numpy", &calculate_iou_matrix_numpy,
          "Compute NxN IoU Matrix on GPU, return as NumPy array",
          py::arg("ellipses"), py::arg("num_points") = 16);

    m.def("calculate_iou_matrix_numpy_from_numpy", &calculate_iou_matrix_numpy_from_numpy,
          "Compute NxN IoU Matrix on GPU from NumPy array (N,5)",
          py::arg("ellipses"), py::arg("num_points") = 16);

    // rectangular Matrix
    m.def("calculate_iou_rectangular", &calculate_iou_rectangular,
          "Compute (rows x cols) IoU Matrix on GPU from two lists",
          py::arg("rows"), py::arg("cols"), py::arg("num_points") = 16);

    m.def("calculate_iou_rectangular_numpy", &calculate_iou_rectangular_numpy,
          "Compute (rows x cols) IoU Matrix on GPU, return as NumPy array",
          py::arg("rows"), py::arg("cols"), py::arg("num_points") = 16);

    m.def("calculate_iou_rectangular_numpy_from_numpy", &calculate_iou_rectangular_numpy_from_numpy,
          "Compute (rows x cols) IoU Matrix from two NumPy arrays (N,5)",
          py::arg("rows"), py::arg("cols"), py::arg("num_points") = 16);

    // batched rectangular matrix
    m.def("calculate_iou_batched_rectangular", &calculate_iou_batched_rectangular,
          "Compute IoU for multiple (det, gt) pairs in a single GPU call.\n\n"
          "This is the OPTIMAL method for detector evaluation - processes all\n"
          "images in a single kernel launch, minimizing GPU overhead.\n\n"
          "Args:\n"
          "  all_dets: (N_total_dets, 5) - all detections concatenated\n"
          "  all_gts: (M_total_gts, 5) - all ground truths concatenated\n"
          "  pair_info: (num_pairs, 5) - [det_offset, gt_offset, out_offset, n_det, n_gt] per pair\n"
          "  num_points: polygon approximation points (default: 16)\n\n"
          "Returns: (flat_results, total_size)",
          py::arg("all_dets"), py::arg("all_gts"), py::arg("pair_info"), py::arg("num_points") = 16);

    // binary file interface
    m.def("calculate_iou_matrix_from_file", &calculate_iou_matrix_from_file,
          "Compute NxN IoU Matrix from binary file",
          py::arg("filename"), py::arg("num_points") = 16);
}
