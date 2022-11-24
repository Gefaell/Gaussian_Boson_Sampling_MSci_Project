import strawberryfields as sf
from strawberryfields.apps import sample
from strawberryfields.apps import data
import numpy as np
#%% Generate samples from a random 4-dimensional symmetrix matrix
modes = 4
n_mean = 6
samples = 5

A = np.random.normal(0, 1, (modes, modes))
A = A + A.T

s_thresh = sample.sample(A, n_mean, samples, threshold=True)
s_pnr = sample.sample(A, n_mean, samples, threshold=False)

print(s_thresh)
print(s_pnr)
#%%Use preset XANADU samples GBS Datasets
a = data.Planted()