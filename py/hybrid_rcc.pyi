import numpy as np

class SamplingAlgorithm:
  PFR = 0
  SIS = 1

class SamplingOutput:
  sample_opt: np.array
  sample_index: int = 0
  total_number_samples: int = 0
  seed: int = 0
  signal: np.array
  box_dimensions: np.array

  def __init__(self, np.array, int, int, int, np.array, np.array): ...
  def __str__(self) -> str:


def sample_gaussian(q_mean: np.array, q_std: np.array, p_mean: np.array,
                               p_std: np.array,
                               sampling_algorithm: SamplingAlgorithm,
                               seed: int, N_max: int,
                               verbose: bool) -> SamplingOutput: ...

def sample_gaussian_hybrid(q_mean: np.array, q_std: np.array, p_mean: np.array,
                               p_std: np.array,
                               sampling_algorithm: SamplingAlgorithm, eps: float,
                               seed: int, N_max: int,
                               verbose: bool) -> SamplingOutput: ...



def decode_gaussian_hybrid(h: SamplingOutput, p_mean: np.array, p_std: np.array) -> np.array: ...