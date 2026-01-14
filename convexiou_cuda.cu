#include "device_iou.cuh"
#include <iostream>
#include <vector>
#include <algorithm>

// polygon points may be changed to decrease time/increase accuracy
#define DEFAULT_NUM_POINTS 16
#define TILE_SIZE 16
#define CHUNK_SIZE 4096

struct EllipseData
{
    double x, y, w, h, angle_rad;
};

// PART 1: PAIR per PAIR IoU

__global__ void compute_iou_kernel(const EllipseData *ellipses1, const EllipseData *ellipses2, double *results, int n, int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    EllipseData e1 = ellipses1[idx];
    EllipseData e2 = ellipses2[idx];

    double a1 = e1.w / sqrt(M_PI);
    double b1 = e1.h / sqrt(M_PI);
    double a2 = e2.w / sqrt(M_PI);
    double b2 = e2.h / sqrt(M_PI);

    DeviceIOU::Polygon poly1, poly2;
    DeviceIOU::approximate_ellipse(e1.x, e1.y, a1, b1, e1.angle_rad, num_points, poly1);
    DeviceIOU::approximate_ellipse(e2.x, e2.y, a2, b2, e2.angle_rad, num_points, poly2);

    double area1 = DeviceIOU::area(poly1);
    double area2 = DeviceIOU::area(poly2);

    DeviceIOU::Polygon interPoly;
    DeviceIOU::compute_intersection_polygon(poly1, poly2, interPoly);
    double intersection = DeviceIOU::area(interPoly);

    double union_area = area1 + area2 - intersection;
    results[idx] = (union_area > EPS) ? (intersection / union_area) : 0.0;
}

extern "C" void run_iou_cuda(const std::vector<EllipseData> &h_e1,
                             const std::vector<EllipseData> &h_e2,
                             std::vector<double> &h_results,
                             int num_points)
{
    if (num_points <= 0)
        num_points = DEFAULT_NUM_POINTS;

    int n = h_e1.size();
    h_results.resize(n);

    EllipseData *d_e1, *d_e2;
    double *d_results;
    size_t size_bytes = n * sizeof(EllipseData);
    size_t res_size_bytes = n * sizeof(double);

    cudaMalloc(&d_e1, size_bytes);
    cudaMalloc(&d_e2, size_bytes);
    cudaMalloc(&d_results, res_size_bytes);

    cudaMemcpy(d_e1, h_e1.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e2, h_e2.data(), size_bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    compute_iou_kernel<<<numBlocks, blockSize>>>(d_e1, d_e2, d_results, n, num_points);

    cudaDeviceSynchronize();

    cudaMemcpy(h_results.data(), d_results, res_size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_e1);
    cudaFree(d_e2);
    cudaFree(d_results);
}

// 2. NxN MATRIX

__global__ void iou_matrix_kernel_tiled(const EllipseData *d_ellipses_row,
                                        const EllipseData *d_ellipses_col,
                                        float *matrix,
                                        int num_rows, int num_cols, int num_points)
{
    __shared__ EllipseData s_rows[TILE_SIZE];
    __shared__ EllipseData s_cols[TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_idx = blockIdx.y * TILE_SIZE + ty;
    int col_idx = blockIdx.x * TILE_SIZE + tx;

    // loading into shared memory
    if (ty < TILE_SIZE && row_idx < num_rows)
        s_rows[ty] = d_ellipses_row[row_idx];
    if (tx < TILE_SIZE && col_idx < num_cols)
        s_cols[tx] = d_ellipses_col[col_idx];

    __syncthreads();

    if (row_idx < num_rows && col_idx < num_cols)
    {
        EllipseData e1 = s_rows[ty];
        EllipseData e2 = s_cols[tx];

        double a1 = e1.w / sqrt(M_PI);
        double b1 = e1.h / sqrt(M_PI);
        double a2 = e2.w / sqrt(M_PI);
        double b2 = e2.h / sqrt(M_PI);

        DeviceIOU::Polygon poly1, poly2;
        DeviceIOU::approximate_ellipse(e1.x, e1.y, a1, b1, e1.angle_rad, num_points, poly1);
        DeviceIOU::approximate_ellipse(e2.x, e2.y, a2, b2, e2.angle_rad, num_points, poly2);

        double area1 = DeviceIOU::area(poly1);
        double area2 = DeviceIOU::area(poly2);

        DeviceIOU::Polygon interPoly;
        DeviceIOU::compute_intersection_polygon(poly1, poly2, interPoly);
        double intersection = DeviceIOU::area(interPoly);

        double union_area = area1 + area2 - intersection;
        matrix[row_idx * num_cols + col_idx] = (float)((union_area > EPS) ? (intersection / union_area) : 0.0);
    }
}

extern "C" void compute_iou_matrix_cuda(const std::vector<EllipseData> &ellipses,
                                        std::vector<float> &h_matrix,
                                        int num_points)
{
    if (num_points <= 0)
        num_points = DEFAULT_NUM_POINTS;

    int n = ellipses.size();
    if (n == 0)
        return;
    h_matrix.assign((size_t)n * n, 0.0f);

    EllipseData *d_ellipses_all;
    float *d_chunk_matrix;

    cudaMalloc(&d_ellipses_all, n * sizeof(EllipseData));
    cudaMalloc(&d_chunk_matrix, (size_t)CHUNK_SIZE * CHUNK_SIZE * sizeof(float));

    cudaMemcpy(d_ellipses_all, ellipses.data(), n * sizeof(EllipseData), cudaMemcpyHostToDevice);

    // chunking loop for processing large matrices in parts
    for (int i = 0; i < n; i += CHUNK_SIZE)
    {
        int rows_in_chunk = std::min(CHUNK_SIZE, n - i);
        for (int j = 0; j < n; j += CHUNK_SIZE)
        {
            int cols_in_chunk = std::min(CHUNK_SIZE, n - j);

            dim3 blockSize(TILE_SIZE, TILE_SIZE);
            dim3 numBlocks((cols_in_chunk + TILE_SIZE - 1) / TILE_SIZE,
                           (rows_in_chunk + TILE_SIZE - 1) / TILE_SIZE);

            iou_matrix_kernel_tiled<<<numBlocks, blockSize>>>(
                d_ellipses_all + i, d_ellipses_all + j,
                d_chunk_matrix, rows_in_chunk, cols_in_chunk, num_points);

            cudaDeviceSynchronize();

            for (int r = 0; r < rows_in_chunk; ++r)
            {
                size_t host_offset = (size_t)(i + r) * n + j;
                size_t device_offset = (size_t)r * cols_in_chunk;
                cudaMemcpy(&h_matrix[host_offset], &d_chunk_matrix[device_offset],
                           cols_in_chunk * sizeof(float), cudaMemcpyDeviceToHost);
            }
        }
    }

    cudaFree(d_ellipses_all);
    cudaFree(d_chunk_matrix);
}

// 3. RECTANGULAR MATRIX (PER IMAGE)

__global__ void iou_rect_kernel(const EllipseData *d_rows,
                                const EllipseData *d_cols,
                                float *matrix,
                                int num_rows, int num_cols, int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * num_cols;
    if (idx >= total)
        return;

    int row_idx = idx / num_cols;
    int col_idx = idx % num_cols;

    EllipseData e1 = d_rows[row_idx];
    EllipseData e2 = d_cols[col_idx];

    double a1 = e1.w / sqrt(M_PI);
    double b1 = e1.h / sqrt(M_PI);
    double a2 = e2.w / sqrt(M_PI);
    double b2 = e2.h / sqrt(M_PI);

    DeviceIOU::Polygon poly1, poly2;
    DeviceIOU::approximate_ellipse(e1.x, e1.y, a1, b1, e1.angle_rad, num_points, poly1);
    DeviceIOU::approximate_ellipse(e2.x, e2.y, a2, b2, e2.angle_rad, num_points, poly2);

    double area1 = DeviceIOU::area(poly1);
    double area2 = DeviceIOU::area(poly2);

    DeviceIOU::Polygon interPoly;
    DeviceIOU::compute_intersection_polygon(poly1, poly2, interPoly);
    double intersection = DeviceIOU::area(interPoly);

    double union_area = area1 + area2 - intersection;
    matrix[row_idx * num_cols + col_idx] = (float)((union_area > EPS) ? (intersection / union_area) : 0.0);
}

extern "C" void compute_iou_rectangular_cuda(const std::vector<EllipseData> &rows,
                                             const std::vector<EllipseData> &cols,
                                             std::vector<float> &h_matrix,
                                             int num_points)
{
    if (num_points <= 0)
        num_points = DEFAULT_NUM_POINTS;

    int num_rows = rows.size();
    int num_cols = cols.size();
    if (num_rows == 0 || num_cols == 0)
    {
        h_matrix.clear();
        return;
    }

    h_matrix.assign((size_t)num_rows * num_cols, 0.0f);

    EllipseData *d_rows = nullptr;
    EllipseData *d_cols = nullptr;
    float *d_matrix = nullptr;

    size_t rows_bytes = num_rows * sizeof(EllipseData);
    size_t cols_bytes = num_cols * sizeof(EllipseData);
    size_t mat_bytes = (size_t)num_rows * num_cols * sizeof(float);

    cudaMalloc(&d_rows, rows_bytes);
    cudaMalloc(&d_cols, cols_bytes);
    cudaMalloc(&d_matrix, mat_bytes);

    cudaMemcpy(d_rows, rows.data(), rows_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), cols_bytes, cudaMemcpyHostToDevice);

    int total = num_rows * num_cols;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    iou_rect_kernel<<<numBlocks, blockSize>>>(d_rows, d_cols, d_matrix, num_rows, num_cols, num_points);
    cudaDeviceSynchronize();

    cudaMemcpy(h_matrix.data(), d_matrix, mat_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_matrix);
}

// 4. BATCHED RECTANGULAR IoU

__global__ void iou_batched_rect_kernel(
    const EllipseData *d_all_dets,
    const EllipseData *d_all_gts,
    float *d_all_results,
    const int *d_pair_info, // [det_offset, gt_offset, out_offset, n_det, n_gt] per pair
    int num_pairs,
    int total_computations,
    int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_computations)
        return;

    // binary search to find which pair this thread belongs to
    int pair_idx = 0;
    int cumsum = 0;
    for (int p = 0; p < num_pairs; ++p)
    {
        int n_det = d_pair_info[p * 5 + 3];
        int n_gt = d_pair_info[p * 5 + 4];
        int pair_size = n_det * n_gt;
        if (idx < cumsum + pair_size)
        {
            pair_idx = p;
            break;
        }
        cumsum += pair_size;
    }

    int det_offset = d_pair_info[pair_idx * 5 + 0];
    int gt_offset = d_pair_info[pair_idx * 5 + 1];
    int out_offset = d_pair_info[pair_idx * 5 + 2];
    int n_det = d_pair_info[pair_idx * 5 + 3];
    int n_gt = d_pair_info[pair_idx * 5 + 4];

    // recompute for pair_idx to get local index
    cumsum = 0;
    for (int p = 0; p < pair_idx; ++p)
    {
        cumsum += d_pair_info[p * 5 + 3] * d_pair_info[p * 5 + 4];
    }
    int local_idx = idx - cumsum;

    int row_idx = local_idx / n_gt;
    int col_idx = local_idx % n_gt;

    if (row_idx >= n_det || col_idx >= n_gt)
        return;

    EllipseData e1 = d_all_dets[det_offset + row_idx];
    EllipseData e2 = d_all_gts[gt_offset + col_idx];

    double a1 = e1.w / sqrt(M_PI);
    double b1 = e1.h / sqrt(M_PI);
    double a2 = e2.w / sqrt(M_PI);
    double b2 = e2.h / sqrt(M_PI);

    DeviceIOU::Polygon poly1, poly2;
    DeviceIOU::approximate_ellipse(e1.x, e1.y, a1, b1, e1.angle_rad, num_points, poly1);
    DeviceIOU::approximate_ellipse(e2.x, e2.y, a2, b2, e2.angle_rad, num_points, poly2);

    double area1 = DeviceIOU::area(poly1);
    double area2 = DeviceIOU::area(poly2);

    DeviceIOU::Polygon interPoly;
    DeviceIOU::compute_intersection_polygon(poly1, poly2, interPoly);
    double intersection = DeviceIOU::area(interPoly);

    double union_area = area1 + area2 - intersection;
    d_all_results[out_offset + row_idx * n_gt + col_idx] =
        (float)((union_area > EPS) ? (intersection / union_area) : 0.0);
}

extern "C" void compute_iou_batched_rectangular_cuda(
    const std::vector<EllipseData> &all_dets,
    const std::vector<EllipseData> &all_gts,
    const std::vector<int> &pair_info, // [det_off, gt_off, out_off, n_det, n_gt] × num_pairs
    std::vector<float> &h_results,
    int total_output_size,
    int num_points)
{
    if (num_points <= 0)
        num_points = DEFAULT_NUM_POINTS;

    int num_pairs = pair_info.size() / 5;
    if (num_pairs == 0 || all_dets.empty())
    {
        h_results.clear();
        return;
    }

    // compute IoU total
    long long total_computations = 0;
    for (int p = 0; p < num_pairs; ++p)
    {
        total_computations += (long long)pair_info[p * 5 + 3] * pair_info[p * 5 + 4];
    }

    h_results.assign(total_output_size, 0.0f);

    EllipseData *d_all_dets = nullptr;
    EllipseData *d_all_gts = nullptr;
    float *d_results = nullptr;
    int *d_pair_info = nullptr;

    cudaMalloc(&d_all_dets, all_dets.size() * sizeof(EllipseData));
    cudaMalloc(&d_all_gts, all_gts.size() * sizeof(EllipseData));
    cudaMalloc(&d_results, total_output_size * sizeof(float));
    cudaMalloc(&d_pair_info, pair_info.size() * sizeof(int));

    cudaMemcpy(d_all_dets, all_dets.data(), all_dets.size() * sizeof(EllipseData), cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_gts, all_gts.data(), all_gts.size() * sizeof(EllipseData), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pair_info, pair_info.data(), pair_info.size() * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (total_computations + blockSize - 1) / blockSize;

    iou_batched_rect_kernel<<<numBlocks, blockSize>>>(
        d_all_dets, d_all_gts, d_results, d_pair_info,
        num_pairs, (int)total_computations, num_points);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results.data(), d_results, total_output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_all_dets);
    cudaFree(d_all_gts);
    cudaFree(d_results);
    cudaFree(d_pair_info);
}
