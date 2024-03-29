#%%
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from greedy import Greedy
from scipy.stats import unitary_group
from gbs_simulation import GBS_simulation
import strawberryfields as sf
from strawberryfields import ops
from utils import total_variation_distance, kl_divergence
from tqdm import tqdm
from utils import bitstring_to_int, convert_to_clicks
import itertools as iter
from itertools import combinations

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 50

n_modes = 6
cutoff = 5
k_order = 2
L = 1200
unitary = unitary_group.rvs(n_modes, random_state=1)

def ideal_renema_circuit_10(n_modes: int, unitary: np.ndarray):
    """Boson sampler with single photons as inputs in half of the modes
    (in the upper half) and vacuum states in the lower half."""
    num_non_vacuum_modes = int(np.ceil(0.5*n_modes))
    index = np.zeros((n_modes,), dtype=np.int16)
    index[:num_non_vacuum_modes] = 1
    ket = np.zeros([cutoff]*n_modes, dtype=np.complex128)
    ket[tuple(index)] = 1.0 + 1j*0.0
    prog = sf.Program(n_modes)
    with prog.context as q:
        ops.Ket(ket) | q
        ops.Interferometer(unitary) | q
    return prog

def ideal_renema_circuit_01(n_modes: int, unitary: np.ndarray):
    """Boson sampler with single photons as inputs in half of the modes
    (in the lower half) and vacuum states in the upper half."""
    num_non_vacuum_modes = int(np.ceil(0.5*n_modes))
    index = np.zeros((n_modes,), dtype=np.int16)
    index[num_non_vacuum_modes:] = 1
    ket = np.zeros([cutoff]*n_modes, dtype=np.complex128)
    ket[tuple(index)] = 1.0 + 1j*0.0
    prog = sf.Program(n_modes)
    with prog.context as q:
        ops.Ket(ket) | q
        ops.Interferometer(unitary) | q
    return prog

def semi_renema_circuit_10(n_modes: int, unitary: np.ndarray):
    """Boson sampler with single photons as inputs in half of the modes
    (in the upper half) and vacuum states in the lower half."""
    num_non_vacuum_modes = int(np.ceil(0.5*n_modes))
    index = np.zeros((n_modes,), dtype=np.int16)
    index[:num_non_vacuum_modes] = 1
    ket = np.zeros([cutoff]*n_modes, dtype=np.complex128)
    ket[tuple(index)] = 1.0 + 1j*0.0
    prog = sf.Program(n_modes)
    with prog.context as q:
        ops.Ket(ket) | q
        ops.Interferometer(unitary) | q
    return prog

def semi_renema_circuit_01(n_modes: int, unitary: np.ndarray):
    """Boson sampler with single photons as inputs in half of the modes
    (in the lower half) and vacuum states in the upper half."""
    num_non_vacuum_modes = int(np.ceil(0.5*n_modes))
    index = np.zeros((n_modes,), dtype=np.int16)
    index[1:num_non_vacuum_modes+1] = 1
    ket = np.zeros([cutoff]*n_modes, dtype=np.complex128)
    ket[tuple(index)] = 1.0 + 1j*0.0
    prog = sf.Program(n_modes)
    with prog.context as q:
        ops.Ket(ket) | q
        ops.Interferometer(unitary) | q
    return prog

def noisy_renema_circuit_10(n_modes: int, unitary: np.ndarray, loss: float):
    """Boson sampler with single photons as inputs in half of the modes
    (in the upper half) and vacuum states in the lower half. It also
    incorporates optical loss with the same beamsplitter construction used
    in the gbs_simulation.py file."""
    loss = loss*np.pi/2
    num_non_vacuum_modes = int(np.ceil(0.5*n_modes))
    index = np.zeros((2*n_modes,), dtype=np.int16)
    index[:num_non_vacuum_modes] = 1
    ket = np.zeros([cutoff]*2*n_modes, dtype=np.complex128)
    ket[tuple(index)] = 1.0 + 1j*0.0
    prog = sf.Program(2*n_modes)
    with prog.context as q:
        ops.Ket(ket) | q
        for cmd in ops.Interferometer(unitary).decompose(tuple([qubit for qubit in q[:n_modes]])):
            cmd.op | cmd.reg
        for i, qubit in enumerate(q[:n_modes]):
            ops.BSgate(loss) | (qubit, q[n_modes + i])
    return prog

def noisy_renema_circuit_01(n_modes: int, unitary: np.ndarray, loss: float):
    """Boson sampler with single photons as inputs in half of the modes
    (in the lower half) and vacuum states in the upper half. It also
    incorporates optical loss with the same beamsplitter construction used
    in the gbs_simulation.py file."""
    loss = loss*np.pi/2
    num_non_vacuum_modes = int(np.ceil(0.5*n_modes))
    index = np.zeros((2*n_modes,), dtype=np.int16)
    index[num_non_vacuum_modes:] = 1
    ket = np.zeros([cutoff]*2*n_modes, dtype=np.complex128)
    ket[tuple(index)] = 1.0 + 1j*0.0
    prog = sf.Program(2*n_modes)
    with prog.context as q:
        ops.Ket(ket) | q
        for cmd in ops.Interferometer(unitary).decompose(tuple([qubit for qubit in q[:n_modes]])):
            cmd.op | cmd.reg
        for i, qubit in enumerate(q[:n_modes]):
            ops.BSgate(loss) | (qubit, q[n_modes + i])
    return prog

def get_state_vector_from_program(
    program,
    fock_cutoff: int
) -> np.ndarray:
    """Runs a Strawberry Fields program in the Fock backend (with the specified cutoff)
    and obtains the output stave vector."""
    eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff})
    result = eng.run(program)
    fock_ket = result.state.ket()
    return fock_ket

def get_fock_prob(
    state_vec: np.ndarray,
    modes: List, 
    photon_numbers: Tuple[int, ...]
) -> float:
    """Returns the probability of detecting a specific photon pattern in the specified
    modes from the output state vector of a GBS simulation."""
    return np.sum([(x*x.conjugate()).real for i, x in np.ndenumerate(state_vec) if tuple([i[j] for j in modes]) == photon_numbers])

def get_threshold_marginal_from_statevec(
    ket: np.ndarray,
    target_modes: List
) -> List:
    """Returns the threshold marginal distribution (calculated from the state vector)
    of the specified target modes."""
    fock_cutoff = ket.shape[0]
    outcomes = [p for p in iter.product(list(range(fock_cutoff)), repeat = len(target_modes))]
    marginal = [get_fock_prob(ket, target_modes, n) for n in outcomes]
    clicks = [bitstring_to_int(x) for x in convert_to_clicks(outcomes)]
    inds_to_sum = [[i for i, x in enumerate(clicks) if x == j] for j in range(2**len(target_modes))]
    threshold_marginal = [np.sum([p for i, p in enumerate(marginal) if i in inds]) for inds in inds_to_sum]
    return threshold_marginal

def get_all_ideal_marginals_from_statevec(
    n_modes: int,
    ket: np.ndarray,
    k_order: int
) -> np.ndarray:
    """Returns all kth-order marginal distributions (calculated from the state vector)."""
    comb = [list(c) for c in combinations(list(range(n_modes)), k_order)]
    marginals : List = []
    for modes in comb:
        marg = get_threshold_marginal_from_statevec(ket, modes)
        marginals.append([modes, marg])
    return np.array(marginals)

def get_renema_output_statevec(n_modes, unitary, cutoff):
    """Superposes the output state vectors of the two disjunct Renema circuits and returns the
    corresponding state vector."""
    ket1 = get_state_vector_from_program(ideal_renema_circuit_01(n_modes, unitary), cutoff)
    ket2 = get_state_vector_from_program(ideal_renema_circuit_10(n_modes, unitary), cutoff)
    superposition_ket = (1/np.sqrt(2))*(ket1 + ket2)
    return superposition_ket

def get_semi_renema_output_statevec(n_modes, unitary, cutoff):
    """Superposes the output state vectors of the two disjunct Renema circuits and returns the
    corresponding state vector."""
    ket1 = get_state_vector_from_program(semi_renema_circuit_01(n_modes, unitary), cutoff)
    ket2 = get_state_vector_from_program(semi_renema_circuit_10(n_modes, unitary), cutoff)
    superposition_ket = (1/np.sqrt(2))*(ket1 + ket2)
    return superposition_ket

def get_noisy_renema_output_statevec(n_modes, unitary, cutoff, loss):
    """Superposes the output state vectors of the two disjunct Renema circuits (incorporating
    optical loss) and returns the corresponding state vector."""
    ket1 = get_state_vector_from_program(noisy_renema_circuit_01(n_modes, unitary, loss), cutoff)
    ket2 = get_state_vector_from_program(noisy_renema_circuit_10(n_modes, unitary, loss), cutoff)
    superposition_ket = (1/np.sqrt(2))*(ket1 + ket2)
    return superposition_ket

#%%
gbs = GBS_simulation()
greedy = Greedy()

# ideal_margs = get_all_ideal_marginals_from_statevec(n_modes, get_renema_output_statevec(n_modes, unitary, cutoff), k_order)
# greedy_matrix = greedy.get_S_matrix(n_modes, L, k_order, ideal_margs)
# greedy_distr = greedy.get_distribution_from_outcomes(greedy_matrix)
# np.save(f'greedy_renema_distr_n={n_modes}_cut={cutoff}_L={L}', greedy_distr)
# ket = get_renema_output_statevec(n_modes, unitary, cutoff)
# ideal_distr = np.array(get_threshold_marginal_from_statevec(ket, list(range(n_modes))))
# np.save(f'ideal_renema_distr_n={n_modes}_cut={cutoff}', ideal_distr)
# distance = total_variation_distance(ideal_distr, greedy_distr)
# print(distance)

# ket = get_semi_renema_output_statevec(n_modes, unitary, cutoff)
# ideal_margs = get_all_ideal_marginals_from_statevec(n_modes, ket, k_order)
# greedy_matrix = greedy.get_S_matrix(n_modes, L, k_order, ideal_margs)
# greedy_distr = greedy.get_distribution_from_outcomes(greedy_matrix)
# np.save(f'greedy_semi_renema_distr_n={n_modes}_cut={cutoff}_L={L}', greedy_distr)
# ideal_distr = np.array(get_threshold_marginal_from_statevec(ket, list(range(n_modes))))
# np.save(f'ideal_semi_renema_distr_n={n_modes}_cut={cutoff}', ideal_distr)
# distance = total_variation_distance(ideal_distr, greedy_distr)
# print(distance)

# x = list(range(2**n_modes))
# plt.plot(x, greedy_distr, label='Greedy')
# plt.plot(x, ideal_distr, label='Ideal')
# plt.xlabel('Bitstring')
# plt.ylabel('Probability')
# plt.legend()
# plt.show()


#%%
# loss = np.linspace(0, 1, 8)

# distances = []
# for i in tqdm(loss):
#     ket = get_noisy_renema_output_statevec(n_modes, unitary, cutoff, i)
#     ideal_distr = np.array(get_threshold_marginal_from_statevec(ket, list(range(n_modes))))
#     print(ideal_distr)
#     distance = total_variation_distance(ideal_distr, greedy_distr)
#     distances.append(distance)

# plt.plot(loss, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
# plt.xlabel('Loss')
# plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
# plt.legend()
# plt.show()

#%%
renema_distr = np.load('ideal_renema_distr_n=6_cut=5.npy')
greedy_renema_distr = np.load('greedy_renema_distr_n=6_cut=5_L=1200.npy')
semi_renema_distr = np.load('ideal_semi_renema_distr_n=6_cut=5.npy')
greedy_semi_renema_distr = np.load('greedy_semi_renema_distr_n=6_cut=5_L=1200.npy')
bs_distr = np.load('ideal_bs_distr_n=6_cut=5_n_input=3.npy')
greedy_bs_distr = np.load('greedy_bs_distr_n=6_cut=5_L=1200_n_input=3.npy')

dist_renema = total_variation_distance(renema_distr, greedy_renema_distr)
dist_semi = total_variation_distance(semi_renema_distr, greedy_semi_renema_distr)
dist_bs = total_variation_distance(bs_distr, greedy_bs_distr)
dists = [dist_renema, dist_semi, dist_bs]

print(dist_renema)
print(dist_semi)
print(dist_bs)
print(

)

div_renema = kl_divergence(renema_distr, greedy_renema_distr)
div_semi = kl_divergence(semi_renema_distr, greedy_semi_renema_distr)
div_bs = kl_divergence(bs_distr, greedy_bs_distr)
divs = [div_renema, div_semi, div_bs]

print(div_renema)
print(div_semi)
print(div_bs)


with plt.style.context(['science']):
    fig, ax = plt.subplots(figsize=[8,6])
    ax2 = ax.twinx()
    ax.bar(np.array([0,1,2])-0.15, dists, width = 0.3, color='royalblue', label='Total Variation Distance')
    ax2.bar(np.array([0,1,2])+0.15, divs, width = 0.3, color='skyblue', label='KL Divergence')
    ax.set_ylabel(r'$\mathcal{D}$(Greedy,Ground)', fontsize=20)
    ax2.set_ylabel(r'$\mathcal{KL}$(Greedy,Ground)', fontsize=20)
    ax.set_xticks(range(3), ['R', 'S-R', 'BS'], size=16)
    ax.set_yticks(np.round(np.linspace(0, dist_renema +0.03, 5), 2))
    ax2.set_yticks(np.round(np.linspace(0, div_renema +0.03, 5), 2))
    ax.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)
    ax.legend(fontsize=16, loc='upper left')
    ax2.legend(fontsize=16, loc='upper right')
    plt.tight_layout()
    plt.savefig('renema_states_histogram.png', dpi=600)
plt.show()


