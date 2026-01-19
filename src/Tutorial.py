from hsmmlearn.hsmm import GaussianHSMM
import numpy as np

durations = np.zeros((3, 10)) # XXXX
durations[:, :] = 0.05
durations[0, 1] = durations[1, 5] = durations[2, 9] = 0.55

tmat = np.array([
    [0.0, 0.5, 0.5],
    [0.3, 0.0, 0.7],
    [0.6, 0.4, 0.0]
])

means = np.array([0.0, 5.0, 10.0])
scales = np.ones_like(means)

hsmm = GaussianHSMM(
    means, scales, durations, tmat,
)

observations, states = hsmm.sample(200)

equal_prob_durations = np.full((3, 10), 0.1)
new_hsmm = GaussianHSMM(
    means, scales, equal_prob_durations, tmat,
)

(is_converged, log_likelihood) = new_hsmm.fit(observations)
print(is_converged, log_likelihood)