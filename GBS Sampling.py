#%%
import strawberryfields as sf
from strawberryfields.apps import sample
from strawberryfields.apps import data
import numpy as np
from utils import bitstring_to_int
from collections import Counter
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
'''
Specifics of data.Planted():

n_mean = 8
threshold = True
n_samples = 50000
modes = 30

so a here is a 50000x30 matrix (each row is 1 sample)

'''
a = np.array(data.Planted())
#print(a[0,:])
#%%
def get_distribution_from_outcomes(samples) -> np.ndarray:
        """Turns list of outcomes (bitstrings) into empirical distribution."""
        bitstrings = [tuple(x) for x in samples]
        sorted_decimal_list = np.sort([bitstring_to_int(binary) for binary in bitstrings])
        count_dict = Counter(sorted_decimal_list)
        counts = [count_dict.get(i, 0) for i in range(2**len(samples[0]))]
        distribution = np.array(counts) / np.sum(counts)
        return distribution
#%%

def marginals_from_samples(a, R):
    '''
    a: np.array that contains all the samples, each row = 1 sample

    R: list with the modes to get the martginals from    
    '''
    marg_col = a[:,R] #marginal columns
    samples = [marg_col[i,:] for i in range(len(marg_col[:,0]))]
    distribution = get_distribution_from_outcomes(samples)
    return distribution


empirical_marginals = marginals_from_samples(a,[0,3,5,6])