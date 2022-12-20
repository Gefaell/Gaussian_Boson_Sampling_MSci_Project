import numpy as np
import matplotlib.pyplot as plt
from greedy import Greedy

#%% Test greedy algorithm

marginals = np.array(
    [[
        [[0,1], [0.25, 0.25, 0.25, 0.25]]
     ],
     [
        [[0,2], [0.25, 0.25, 0.25, 0.25]],
        [[1,2], [0.25, 0.25, 0.25, 0.25]]
     ],
     [
        [[0,3], [0.25, 0.25, 0.25, 0.25]],
        [[1,3], [0.25, 0.25, 0.25, 0.25]],
        [[2,3], [0.25, 0.25, 0.25, 0.25]]
     ]])
n_modes = 4
k_order = 2

print(Greedy().get_S_matrix(n_modes, 20, k_order, marginals))
