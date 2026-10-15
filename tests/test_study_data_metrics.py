import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from src.study.engine import report
from src.study.data import prepare
from src.study.geometry import exp
from src.study.runtime import completion_metrics, denoising_metrics, temporal


class StudyMetricTests(unittest.TestCase):
    def test_temporal_loss_does_not_bridge_segments(self):
        first = exp(torch.zeros(60, 21, 3))
        second = exp(torch.ones(60, 21, 3))
        self.assertLess(float(temporal(torch.cat((first, second)), segments=2)), 1e-5)
        self.assertGreater(float(temporal(torch.cat((first, second)))), 0)

    def test_manifest_reproducibility_contiguity_and_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'data/3dpw'
            data.mkdir(parents=True)
            for split in ('train', 'validation', 'test'):
                n = 240
                np.savez(data/f'{split}.npz', poses=np.zeros((n, 63), dtype=np.float32),
                    sequence=np.repeat([split+'_a', split+'_b'], 120), person=np.zeros(n, dtype=int),
                    frame=np.tile(np.arange(120), 2))
            config = {'study': 'test', 'seed': 42, 'development': {'poses': 2, 'segments': 2},
                      'final': {'poses': 2, 'segments': 2}, 'smoke': {'poses': 1, 'segments': 1}}
            first = prepare(root, config, root/'output')
            second = prepare(root, config, root/'output')
            self.assertEqual(first, second)
            for cohort in first['cohorts'].values():
                for segment in cohort['segments']:
                    self.assertEqual(len(segment), 60)
                    np.testing.assert_array_equal(np.diff(segment), np.ones(59))
            with np.load(data/'train.npz') as source:
                corrupted = {key: source[key] for key in source.files}
            corrupted['sequence'] = np.full(len(corrupted['poses']), 'validation_a')
            np.savez(data/'train.npz', **corrupted)
            with self.assertRaisesRegex(AssertionError, 'Sequence leakage'):
                prepare(root, config, root/'other')

    def test_zero_error_and_visible_diagnostic(self):
        class FakeBody:
            vertex_joint = torch.arange(1, 22)
            def __call__(self, rotations):
                points = rotations[..., :, 0]
                return torch.cat((torch.zeros(len(points), 1, 3), points), 1), points
        gt = exp(torch.randn(2, 21, 3)*.1)
        observed = torch.ones(21, dtype=torch.bool)
        observed[[0, 3, 6, 9]] = False
        hypotheses = gt.repeat_interleave(10, 0)
        result, _ = completion_metrics(FakeBody(), hypotheses, gt, observed, 10)
        np.testing.assert_allclose(result['mpvpe_mm'], 0, atol=1e-5)
        np.testing.assert_allclose(result['apd_cm'], 0, atol=1e-5)
        dirty = hypotheses.clone()
        dirty[:, observed] = exp(torch.randn(20, int(observed.sum()), 3))
        result, restored = completion_metrics(FakeBody(), dirty, gt, observed, 10)
        self.assertGreater(np.mean(result['visible_rad']), 0)
        torch.testing.assert_close(restored[:, observed], hypotheses[:, observed])

    def test_denoising_noise_units_and_acceleration(self):
        clean = torch.zeros(60, 22, 3)
        self.assertEqual(denoising_metrics(clean, clean)['mpjpe_mm'], 0)
        noisy = clean.clone()
        noisy[..., 0] = .04
        metrics = denoising_metrics(noisy, clean)
        self.assertAlmostEqual(metrics['mpjpe_mm'], 40, places=3)
        self.assertEqual(metrics['acceleration_error_mm_s2'], 0)

    def test_report_pairs_matching_inference_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = root/'cells'
            cells.mkdir()
            for seed in (0, 1):
                for method, value in [('direct', 100+seed*100), ('dflow', 90+seed*100)]:
                    record = {'status': 'complete', 'cohort': 'development', 'checkpoint_sha256': 'original',
                        'solver': 'intrinsic', 'task': 'denoising', 'condition': '0', 'method': method,
                        'timing': {'seconds': 1, 'peak_allocated_mib': 1},
                        'rows': [{'case': 0, 'seed': seed, 'sequence': 'a', 'mpjpe_mm': value,
                                  'acceleration_error_mm_s2': 0}]}
                    (cells/f'{seed}_{method}.json').write_text(json.dumps(record))
            summary = report(root, {'bootstrap_samples': 20})
            comparison = next(iter(summary['paired_comparisons'].values()))
            self.assertEqual(comparison['mean_difference_mm'], -10.)


if __name__ == '__main__':
    unittest.main()
