import numpy as np
import time

try:
    from convexiou import ellipse_iou, rectangular_iou, batched_iou
    # ellipse_iou is the primary name; rectangular_iou kept for compat
except ImportError:
    print("convexiou not installed. Run: pip install .")
    exit(1)

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.affinity import rotate, translate
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("shapely not installed - will skip CPU comparison. Run: pip install shapely")


def obb_to_shapely_polygon(x, y, w, h, angle_rad):
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    poly = ShapelyPolygon(corners)
    poly = rotate(poly, np.degrees(angle_rad), origin=(0, 0))
    poly = translate(poly, x, y)
    return poly


def shapely_iou(box1, box2):
    p1 = obb_to_shapely_polygon(*box1)
    p2 = obb_to_shapely_polygon(*box2)
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0.0


def generate_test_data(num_images, dets_per_image, gts_per_image, seed=42):
    rng = np.random.RandomState(seed)
    all_dets_list = []
    all_gts_list = []
    for _ in range(num_images):
        dets = rng.rand(dets_per_image, 5) * [800, 600, 100, 100, 2 * np.pi]
        dets[:, 4] -= np.pi
        gts = rng.rand(gts_per_image, 5) * [800, 600, 100, 100, 2 * np.pi]
        gts[:, 4] -= np.pi
        all_dets_list.append(dets.astype(np.float64))
        all_gts_list.append(gts.astype(np.float64))
    return all_dets_list, all_gts_list


def run_gpu_batched(all_dets_list, all_gts_list):
    all_dets = np.vstack(all_dets_list)
    all_gts = np.vstack(all_gts_list)

    pair_info = []
    det_off = gt_off = out_off = 0
    for dets, gts in zip(all_dets_list, all_gts_list):
        nd, ng = len(dets), len(gts)
        pair_info.append([det_off, gt_off, out_off, nd, ng])
        det_off += nd
        gt_off += ng
        out_off += nd * ng
    pair_info = np.array(pair_info, dtype=np.int32)

    _ = batched_iou(all_dets, all_gts, pair_info, num_points=16)

    t0 = time.perf_counter()
    results, total_size = batched_iou(all_dets, all_gts, pair_info, num_points=16)
    gpu_time = time.perf_counter() - t0

    iou_matrices = []
    for i in range(len(all_dets_list)):
        _, _, o_off, nd, ng = pair_info[i]
        iou_matrices.append(results[o_off:o_off + nd * ng].reshape(nd, ng))

    return gpu_time, iou_matrices


def run_shapely_sample(all_dets_list, all_gts_list, sample_images):
    sample_dets = all_dets_list[:sample_images]
    sample_gts = all_gts_list[:sample_images]

    t0 = time.perf_counter()
    iou_matrices = []
    for dets, gts in zip(sample_dets, sample_gts):
        mat = np.zeros((len(dets), len(gts)), dtype=np.float64)
        for i in range(len(dets)):
            for j in range(len(gts)):
                mat[i, j] = shapely_iou(dets[i], gts[j])
        iou_matrices.append(mat)
    sample_time = time.perf_counter() - t0

    return sample_time, iou_matrices


if __name__ == "__main__":
    NUM_IMAGES = 5000
    DETS_PER_IMAGE = 50
    GTS_PER_IMAGE = 10
    SHAPELY_SAMPLE = 5
    total_pairs = NUM_IMAGES * DETS_PER_IMAGE * GTS_PER_IMAGE

    print(f"Test config: {NUM_IMAGES} images, {DETS_PER_IMAGE} dets/img, {GTS_PER_IMAGE} gts/img")
    print(f"Total IoU pairs: {total_pairs:,}")
    print()

    all_dets, all_gts = generate_test_data(NUM_IMAGES, DETS_PER_IMAGE, GTS_PER_IMAGE)

    print("--- GPU Batched (full dataset) ---")
    gpu_time, gpu_matrices = run_gpu_batched(all_dets, all_gts)
    print(f"  Time: {gpu_time*1000:.2f} ms")
    print(f"  Throughput: {total_pairs / gpu_time / 1e6:.2f} M pairs/sec")
    print()

    print("--- GPU Per-Image Rectangular (full dataset) ---")
    _ = rectangular_iou(all_dets[0], all_gts[0], num_points=16)
    t0 = time.perf_counter()
    for dets, gts in zip(all_dets, all_gts):
        rectangular_iou(dets, gts, num_points=16)
    per_image_time = time.perf_counter() - t0
    print(f"  Time: {per_image_time*1000:.2f} ms")
    print(f"  Throughput: {total_pairs / per_image_time / 1e6:.2f} M pairs/sec")
    print(f"  Batched speedup over per-image: {per_image_time / gpu_time:.2f}x")
    print()

    if HAS_SHAPELY:
        print(f"--- Shapely CPU (sample: {SHAPELY_SAMPLE} images) ---")
        sample_pairs = SHAPELY_SAMPLE * DETS_PER_IMAGE * GTS_PER_IMAGE
        shapely_time, shapely_matrices = run_shapely_sample(all_dets, all_gts, SHAPELY_SAMPLE)
        estimated_full = shapely_time / SHAPELY_SAMPLE * NUM_IMAGES
        print(f"  Sample time: {shapely_time*1000:.2f} ms ({sample_pairs:,} pairs)")
        print(f"  Estimated full time: {estimated_full:.2f} sec ({total_pairs:,} pairs)")
        print()

        print("--- Accuracy check (GPU vs Shapely on sample) ---")
        max_diff = 0.0
        mean_diff = 0.0
        count = 0
        for img_i in range(SHAPELY_SAMPLE):
            gpu_mat = gpu_matrices[img_i].astype(np.float64)
            shp_mat = shapely_matrices[img_i]
            diff = np.abs(gpu_mat - shp_mat)
            max_diff = max(max_diff, diff.max())
            mean_diff += diff.sum()
            count += diff.size
        mean_diff /= count
        print(f"  Max absolute difference:  {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        print()

        print("--- Speedup ---")
        print(f"  GPU Batched:       {gpu_time*1000:.2f} ms")
        print(f"  GPU Per-Image:     {per_image_time*1000:.2f} ms")
        print(f"  Shapely estimated: {estimated_full*1000:.2f} ms")
        print(f"  Batched vs Shapely:   {estimated_full / gpu_time:.1f}x")
        print(f"  Per-Image vs Shapely: {estimated_full / per_image_time:.1f}x")
    else:
        print("Shapely not available, skipping CPU comparison.")
