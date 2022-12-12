import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from GBS_marginals import Marginal
from greedy import Greedy
from scipy.stats import unitary_group

marginals = np.array(
    [[0.2, 0.25, 0.3, 0.25],
    [0.35, 0.2, 0.25, 0.2],
    [0.2, 0.3, 0.4, 0.1]])
n_modes = 4
k_order = 2

dists = []
for j in range(20, 140, 40):
    test_matrix = Greedy().get_S_matrix(n_modes, j, k_order, marginals)[0]
    dist_j = Greedy().get_k_mode_marginal_distances_of_greedy_matrix(test_matrix, k_order, marginals)
    dists.append(dist_j)

marginal_index = np.arange(0, len(dists[0]), 1)

plt.figure()
for i in range(len(dists)):
    plt.plot(marginal_index, dists[i], label=f'L = {20 + 40*i}')
plt.xlabel('Submatrix index')
plt.ylabel('Variation distance')
plt.xticks(marginal_index)
plt.legend()
#plt.show()
#plt.savefig('variation_dist.png', dpi=400)

L = 20
distances = Greedy().get_S_matrix(n_modes, L, k_order, marginals)[1]
iteration_number = list(range(len(distances[0])))

plt.figure()
plt.title(f'L = {L}')
for i, d in enumerate(distances):
    plt.plot(iteration_number, d, label=f'Submatrix index = {i}')
plt.xlabel('Iteration number')
plt.ylabel('Variation distance')
plt.xticks(np.arange(0, L, L/5))
plt.legend()

L = 60
distances = Greedy().get_S_matrix(n_modes, L, k_order, marginals)[1]
iteration_number = list(range(len(distances[0])))

plt.figure()
plt.title(f'L = {L}')
for i, d in enumerate(distances):
    plt.plot(iteration_number, d, label=f'Submatrix index = {i}')
plt.xlabel('Iteration number')
plt.ylabel('Variation distance')
plt.xticks(np.arange(0, L, L/5))
plt.legend()

L = 100
distances = Greedy().get_S_matrix(n_modes, L, k_order, marginals)[1]
iteration_number = list(range(len(distances[0])))

plt.figure()
plt.title(f'L = {L}')
for i, d in enumerate(distances):
    plt.plot(iteration_number, d, label=f'Submatrix index = {i}')
plt.xlabel('Iteration number')
plt.ylabel('Variation distance')
plt.xticks(np.arange(0, L, L/5))
plt.legend()
plt.show()

r_k = [
1.6518433645720738,
1.687136454610338,
1.62938385974034,
1.706029877650956,
1.8395638626723685,
1.3943570412472834,
1.4819924169286014,
1.6313980669381827,
1.6559961541267325,
1.3389267197532349,
1.568736620327057,
1.6772334549978614,
1.459031307907052,
1.4124223294979523,
1.3440269631323098,
1.4328684458997072,
1.4675334685180914,
1.6270874674912998,
1.6044404863902908,
1.581538415101846,
1.6519035066626184,
1.5456532234514821,
1.5974577318822245,
1.7043797524114164,
1.7294783286655087]

#r_k = np.array(r_k + r_k)

T_re = pd.read_excel('matrix_re.xlsx', header = None).to_numpy()
T_im = pd.read_excel('matrix_im.xlsx', header = None).to_numpy()
T = T_re + T_im * 1j
T = T.T

ideal_matrix = unitary_group.rvs(50)
#print(Marginal().get_marginal_distribution([1,2], ideal_matrix, r_k))