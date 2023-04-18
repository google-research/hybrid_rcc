import numpy as np
from scipy import stats
import hybrid_rcc


def compare_sampling_outputs(
    a: hybrid_rcc.SamplingOutput, b: hybrid_rcc.SamplingOutput, eps: float
) -> None:
  assert a.seed == b.seed
  assert a.total_number_samples == b.total_number_samples
  assert a.sample_index == b.sample_index
  assert len(a.box_dimensions) == len(b.box_dimensions)
  np.testing.assert_allclose(a.sample_opt, b.sample_opt, atol=eps, rtol=0)

  if len(a.box_dimensions) != 0:  # pylint: disable=g-explicit-length-test
    np.testing.assert_allclose(a.signal, b.signal, atol=eps, rtol=0)
    np.testing.assert_allclose(
        a.box_dimensions, b.box_dimensions, atol=eps, rtol=0
    )


def test_sample_pfr():
  p = stats.norm([0, 0], [0.5, 0.5])
  q = stats.norm([0, 0], [1, 1])
  output = hybrid_rcc.sample_gaussian(
      p.mean(),
      p.std(),
      q.mean(),
      q.std(),
      hybrid_rcc.SamplingAlgorithm.PFR,
      42,
      50,
      False,
  )
  want = hybrid_rcc.SamplingOutput([-0.206789, -0.624549], 0, 3, 42)
  compare_sampling_outputs(output, want, 1e-6)


def test_sample_sis():
  p = stats.norm([0, 0], [0.5, 0.5])
  q = stats.norm([0, 0], [1, 1])
  output = hybrid_rcc.sample_gaussian(
      p.mean(),
      p.std(),
      q.mean(),
      q.std(),
      hybrid_rcc.SamplingAlgorithm.SIS,
      42,
      50,
      False,
  )
  want = hybrid_rcc.SamplingOutput([-0.206789, -0.624549], 0, 3, 42)
  compare_sampling_outputs(output, want, 1e-6)


def test_sample_hybrid_pfr():
  p = stats.norm([1, 2], [2, 0.5])
  q = stats.norm([0, 0], [1, 1.5])
  output = hybrid_rcc.sample_gaussian_hybrid(
      p.mean(),
      p.std(),
      q.mean(),
      q.std(),
      hybrid_rcc.SamplingAlgorithm.PFR,
      1e-4,
      42,
      1000,
      False,
  )
  want = hybrid_rcc.SamplingOutput(
      [-0.229706, 1.465346], 1, 4, 42, np.zeros((2,)), np.ones((2,))
  )
  compare_sampling_outputs(output, want, 1e-6)


def test_sample_hybrid_sis():
  p = stats.norm([0, 0], [0.5, 0.5])
  q = stats.norm([0, 0], [1, 1])
  output = hybrid_rcc.sample_gaussian_hybrid(
      p.mean(),
      p.std(),
      q.mean(),
      q.std(),
      hybrid_rcc.SamplingAlgorithm.SIS,
      0.982,
      42,
      50,
      False,
  )
  want = hybrid_rcc.SamplingOutput(
      [0.074926, 0.047668], 0, 2, 42, np.ones((2,)) * 7, np.ones((2,)) * 14
  )
  compare_sampling_outputs(output, want, 1e-6)


def test_decode_hybrid_pfr():
  p = stats.norm([1, 2], [2, 0.5])
  q = stats.norm([0, 0], [1, 1.5])
  output = hybrid_rcc.sample_gaussian_hybrid(
      p.mean(),
      p.std(),
      q.mean(),
      q.std(),
      hybrid_rcc.SamplingAlgorithm.PFR,
      1e-4,
      42,
      1000,
      False,
  )
  got = hybrid_rcc.decode_gaussian_hybrid(output, p.mean(), p.std())
  np.testing.assert_allclose(got, output.sample_opt, atol=0.5)


def test_decode_hybrid_sis():
  p = stats.norm([0, 0], [0.5, 0.5])
  q = stats.norm([0, 0], [1, 1])
  output = hybrid_rcc.sample_gaussian_hybrid(
      p.mean(),
      p.std(),
      q.mean(),
      q.std(),
      hybrid_rcc.SamplingAlgorithm.SIS,
      0.982,
      42,
      50,
      False,
  )
  got = hybrid_rcc.decode_gaussian_hybrid(output, p.mean(), p.std())
  np.testing.assert_allclose(got, output.sample_opt, atol=0.5)
