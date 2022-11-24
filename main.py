import numpy as np
import matplotlib.pyplot as plt
from greedy import Greedy

marginals = np.array(
    [[0.2, 0.25, 0.3, 0.25],
    [0.35, 0.2, 0.25, 0.2],
    [0.2, 0.3, 0.4, 0.1]])

dists = []
for j in range(20, 180, 40):
    n_modes = 4
    k_order = 2
    test_matrix = Greedy().get_S_matrix(n_modes, j, k_order, marginals)
    final_row_submatrix_indices = [Greedy()._get_submatrix_indices(test_matrix.shape, k_order, i)[-1] for i in range(len(marginals))]
    dist_j = [0.5*np.sum(np.abs(Greedy()._get_marginal_variation_dist(test_matrix, final_row_submatrix_indices[i], marginals[i]))) for i in range(len(marginals))]
    dists.append(dist_j)

marginal_index = np.arange(0, len(dists[0]), 1)

with plt.style.context(['science']):
    for i in range(len(dists)):
        plt.plot(marginal_index, dists[i], label=f'L = {20 + 40*i}')

    plt.xlabel('Marginal index')
    plt.ylabel('Variation distance')
    plt.xticks(marginal_index)
    plt.legend()
    plt.show()
    #plt.savefig('variation_dist.png', dpi=400)
    
#%% testi