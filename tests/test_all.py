#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_all.py -- comprehensive test suite for convexiou (26 tests)

Coverage:
  - Imports & API surface          (tests 01–08)
  - Backward compatibility         (tests 09–10)
  - Bare-array auto-wrap fix       (tests 11–13)
  - List-of-arrays                 (tests 14–16)
  - Edge cases                     (tests 17–20)
  - Gaucho module                  (test  21)
  - GPU accuracy (require CUDA)    (tests 22–26)

Usage:
    python tests/test_all.py
"""

import sys
import os
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Try importing convexiou
# ---------------------------------------------------------------------------
CONVEXIOU_AVAILABLE = False
try:
    import convexiou
    from convexiou import batched_iou_from_lists, batched_iou, matrix_iou
    CONVEXIOU_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Detect whether a real CUDA GPU can execute kernels
# ---------------------------------------------------------------------------
HAS_CUDA_GPU = False
if CONVEXIOU_AVAILABLE:
    try:
        from convexiou import ellipse_iou
        _probe = np.array([[100.0, 100.0, 50.0, 30.0, 0.0]], dtype=np.float64)
        _result = ellipse_iou(_probe, _probe, num_points=16)
        if _result is not None and _result.shape == (1, 1) and float(_result[0, 0]) > 0.5:
            HAS_CUDA_GPU = True
    except Exception:
        pass


# ===================================================================
# Tests 01–08: Import and API surface
# ===================================================================
@unittest.skipUnless(CONVEXIOU_AVAILABLE, "convexiou not installed")
class TestImports(unittest.TestCase):

    def test_01_import_module(self):
        """convexiou module imports and has __version__."""
        self.assertTrue(hasattr(convexiou, '__version__'))

    def test_02_import_ellipse_iou(self):
        """ellipse_iou is importable and callable."""
        from convexiou import ellipse_iou
        self.assertTrue(callable(ellipse_iou))

    def test_03_import_rectangular_iou_compat(self):
        """rectangular_iou still exists (backward compat)."""
        from convexiou import rectangular_iou
        self.assertTrue(callable(rectangular_iou))

    def test_04_import_batched_iou(self):
        """batched_iou is importable."""
        from convexiou import batched_iou
        self.assertTrue(callable(batched_iou))

    def test_05_import_batched_iou_from_lists(self):
        """batched_iou_from_lists is importable."""
        from convexiou import batched_iou_from_lists
        self.assertTrue(callable(batched_iou_from_lists))

    def test_06_import_matrix_iou(self):
        """matrix_iou is importable."""
        from convexiou import matrix_iou
        self.assertTrue(callable(matrix_iou))

    def test_07_all_exports(self):
        """__all__ includes expected primary names."""
        self.assertIn('ellipse_iou', convexiou.__all__)
        self.assertIn('rectangular_iou', convexiou.__all__)
        self.assertIn('batched_iou_from_lists', convexiou.__all__)
        self.assertIn('matrix_iou', convexiou.__all__)
        self.assertIn('batched_iou', convexiou.__all__)

    def test_08_version_string(self):
        """__version__ is a non-empty string."""
        self.assertIsInstance(convexiou.__version__, str)
        self.assertGreater(len(convexiou.__version__), 0)


# ===================================================================
# Tests 09–10: Backward compatibility
# ===================================================================
@unittest.skipUnless(CONVEXIOU_AVAILABLE, "convexiou not installed")
class TestBackwardCompat(unittest.TestCase):

    def test_09_rectangular_iou_is_ellipse_iou(self):
        """rectangular_iou is the same object as ellipse_iou."""
        from convexiou import rectangular_iou, ellipse_iou
        self.assertIs(rectangular_iou, ellipse_iou)

    def test_10_ellipse_iou_callable(self):
        """ellipse_iou is callable."""
        from convexiou import ellipse_iou
        self.assertTrue(callable(ellipse_iou))


# ===================================================================
# Tests 11–13: Bare ndarray auto-wrapping fix
# ===================================================================
@unittest.skipUnless(CONVEXIOU_AVAILABLE, "convexiou not installed")
class TestBareArrayFix(unittest.TestCase):

    def test_11_bare_empty_arrays(self):
        """Bare 2-D arrays with 0 rows should return (0, 0) matrix directly."""
        dets = np.zeros((0, 5), dtype=np.float64)
        gts = np.zeros((0, 5), dtype=np.float64)
        result = batched_iou_from_lists(dets, gts)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0, 0))

    def test_12_bare_empty_dets_nonempty_gts(self):
        """Bare array: 0 dets, 3 gts → (0, 3) matrix directly."""
        dets = np.zeros((0, 5), dtype=np.float64)
        gts = np.zeros((3, 5), dtype=np.float64)
        result = batched_iou_from_lists(dets, gts)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0, 3))

    def test_13_bare_nonempty_dets_empty_gts(self):
        """Bare array: 5 dets, 0 gts → (5, 0) matrix directly."""
        dets = np.zeros((5, 5), dtype=np.float64)
        gts = np.zeros((0, 5), dtype=np.float64)
        result = batched_iou_from_lists(dets, gts)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 0))


# ===================================================================
# Tests 14–16: List-of-arrays behaviour
# ===================================================================
@unittest.skipUnless(CONVEXIOU_AVAILABLE, "convexiou not installed")
class TestListOfArrays(unittest.TestCase):

    def test_14_list_empty_images(self):
        """List of empty-row arrays → list of zero matrices."""
        dets = [np.zeros((0, 5), dtype=np.float64) for _ in range(3)]
        gts = [np.zeros((0, 5), dtype=np.float64) for _ in range(3)]
        result = batched_iou_from_lists(dets, gts)
        self.assertEqual(len(result), 3)
        for mat in result:
            self.assertEqual(mat.shape[0], 0)

    def test_15_list_mixed_empty_nonempty(self):
        """Mix of empty and non-empty arrays; empty ones get zero matrices."""
        dets = [np.zeros((0, 5), dtype=np.float64),
                np.zeros((4, 5), dtype=np.float64)]
        gts = [np.zeros((3, 5), dtype=np.float64),
               np.zeros((0, 5), dtype=np.float64)]
        result = batched_iou_from_lists(dets, gts)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (0, 3))
        self.assertEqual(result[1].shape, (4, 0))

    def test_16_mismatched_list_lengths(self):
        """Mismatched list lengths should raise ValueError."""
        dets = [np.zeros((3, 5), dtype=np.float64)]
        gts = [np.zeros((3, 5), dtype=np.float64),
               np.zeros((2, 5), dtype=np.float64)]
        with self.assertRaises(ValueError):
            batched_iou_from_lists(dets, gts)


# ===================================================================
# Tests 17–20: Edge cases
# ===================================================================
@unittest.skipUnless(CONVEXIOU_AVAILABLE, "convexiou not installed")
class TestEdgeCases(unittest.TestCase):

    def test_17_empty_list(self):
        """Empty lists → empty list."""
        result = batched_iou_from_lists([], [])
        self.assertEqual(result, [])

    def test_18_single_element_empty(self):
        """Single 1-D empty array auto-reshaped to (0, 5)."""
        dets = [np.array([], dtype=np.float64)]
        gts = [np.array([], dtype=np.float64)]
        result = batched_iou_from_lists(dets, gts)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape[0], 0)

    def test_19_float32_coercion(self):
        """float32 input accepted (coerced internally to float64)."""
        dets = [np.zeros((3, 5), dtype=np.float32)]
        gts = [np.zeros((0, 5), dtype=np.float32)]
        result = batched_iou_from_lists(dets, gts)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (3, 0))

    def test_20_matrix_iou_callable(self):
        """matrix_iou is callable."""
        self.assertTrue(callable(matrix_iou))


# ===================================================================
# Test 21: Gaucho module
# ===================================================================
@unittest.skipUnless(CONVEXIOU_AVAILABLE, "convexiou not installed")
class TestGaucho(unittest.TestCase):

    def test_21_gaucho_import(self):
        """gaucho module importable with eval_rbbox_map_gpu."""
        from convexiou import gaucho
        self.assertTrue(hasattr(gaucho, 'eval_rbbox_map_gpu'))
        self.assertTrue(callable(gaucho.eval_rbbox_map_gpu))


# ===================================================================
# Tests 22–26: GPU accuracy (require actual CUDA GPU)
# ===================================================================
@unittest.skipUnless(CONVEXIOU_AVAILABLE, "convexiou not installed")
@unittest.skipUnless(HAS_CUDA_GPU, "Requires CUDA GPU for value checks")
class TestGPUAccuracy(unittest.TestCase):

    def test_22_identical_boxes_iou_near_1(self):
        """Identical boxes → IoU ≈ 1."""
        from convexiou import ellipse_iou
        boxes = np.array([[100, 100, 50, 30, 0.5]], dtype=np.float64)
        iou = ellipse_iou(boxes, boxes, num_points=16)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0, places=2)

    def test_23_non_overlapping_iou_zero(self):
        """Far-apart boxes → IoU = 0."""
        from convexiou import ellipse_iou
        box1 = np.array([[0, 0, 10, 10, 0.0]], dtype=np.float64)
        box2 = np.array([[1000, 1000, 10, 10, 0.0]], dtype=np.float64)
        iou = ellipse_iou(box1, box2, num_points=16)
        self.assertAlmostEqual(float(iou[0, 0]), 0.0, places=5)

    def test_24_ellipse_iou_output_shape(self):
        """ellipse_iou returns correct (N_det, N_gt) shape."""
        from convexiou import ellipse_iou
        np.random.seed(99)
        dets = (np.random.rand(7, 5) * [800, 600, 100, 100, np.pi]).astype(np.float64)
        gts = (np.random.rand(4, 5) * [800, 600, 100, 100, np.pi]).astype(np.float64)
        iou = ellipse_iou(dets, gts, num_points=16)
        self.assertEqual(iou.shape, (7, 4))

    def test_25_bare_array_19x11(self):
        """Colleague's fix: bare (19, 5) and (11, 5) → (19, 11) matrix directly."""
        np.random.seed(42)
        dets = (np.random.rand(19, 5) * [800, 600, 100, 100, np.pi]).astype(np.float64)
        gts = (np.random.rand(11, 5) * [800, 600, 100, 100, np.pi]).astype(np.float64)
        result = batched_iou_from_lists(dets, gts)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (19, 11))
        # Values should be in [0, 1]
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_26_matrix_iou_diagonal(self):
        """matrix_iou diagonal ≈ 1 (self-IoU)."""
        boxes = np.array([
            [100, 100, 50, 30, 0.5],
            [200, 200, 40, 40, 0.0],
            [300, 300, 60, 60, 1.0],
        ], dtype=np.float64)
        iou = matrix_iou(boxes, num_points=16)
        self.assertEqual(iou.shape, (3, 3))
        for i in range(3):
            self.assertAlmostEqual(float(iou[i, i]), 1.0, places=2)


# ===================================================================
if __name__ == '__main__':
    print(f"convexiou available: {CONVEXIOU_AVAILABLE}")
    print(f"CUDA GPU available:  {HAS_CUDA_GPU}")
    print()
    unittest.main(verbosity=2)
