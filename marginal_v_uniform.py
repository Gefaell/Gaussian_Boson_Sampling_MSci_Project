from gbs_simulation import GBS_simulation
import numpy as np
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
from tqdm import tqdm

k_order = 2
n_modes = 3
cutoff = 6
squeezing_params = [0.5] * n_modes
unitary = unitary_group.rvs(n_modes)
target_modes = list(range(0, k_order))
loss = np.linspace(0, np.pi/2, 20)

simul = GBS_simulation()
ground_truth = simul.get_noisy_marginal_from_simul(n_modes, cutoff, squeezing_params, unitary, target_modes, np.pi/2)
print(ground_truth)

distances = []
for angle in tqdm(loss):
    ground_truth = simul.get_noisy_marginal_from_simul(n_modes, cutoff, squeezing_params, unitary, target_modes, angle)
    uniform_distr = np.array([1/(2**k_order)]*(2**k_order))
    distance = 0.5*np.sum(np.abs(ground_truth - uniform_distr))
    distances.append(distance)

plt.figure()
plt.plot(loss, distances)
plt.xlabel('Loss')
plt.ylabel('Distance')

k_order = 3
n_modes = 3
cutoff = 6
squeezing_params = [0.5] * n_modes
unitary = unitary_group.rvs(n_modes)
target_modes = list(range(0, k_order))
loss = np.linspace(0, np.pi/2, 20)

simul = GBS_simulation()
distances = []
for angle in tqdm(loss):
    ground_truth = simul.get_noisy_marginal_from_simul(n_modes, cutoff, squeezing_params, unitary, target_modes, angle)
    uniform_distr = np.array([1/(2**k_order)]*(2**k_order))
    distance = 0.5*np.sum(np.abs(ground_truth - uniform_distr))
    distances.append(distance)

plt.plot(loss, distances)
plt.xlabel('Loss')
plt.ylabel('Distance')
plt.show()


