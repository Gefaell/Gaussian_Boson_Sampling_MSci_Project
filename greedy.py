import copy
from typing import List, Tuple
import numpy as np
from collections import Counter
from utils import bitstring_to_int, int_to_padded_bitstring


class Greedy():

    def _get_submatrix_indices(
        self, 
        shape: Tuple[int, int], 
        k_order: int, 
        iteration_number: int
    ) -> np.ndarray:
        """Return the submatrix indices (tuples) of the submatrix with k columns + L rows
        corresponding to the given iteration number."""
        submatrix_indices = [index for index in np.ndindex(shape) if index[1]
             in list(range(iteration_number, iteration_number + k_order))]
        reshaped_inds = np.empty(len(submatrix_indices), dtype='object')
        reshaped_inds[:] = submatrix_indices
        reshaped_inds = reshaped_inds.reshape(shape[0], k_order)
        return reshaped_inds

    def _get_distribution_from_outcomes(self, samples: np.ndarray) -> np.ndarray:
        """Turns list of outcomes (bitstrings) into empirical distribution."""
        bitstrings = [tuple(x) for x in samples]
        sorted_decimal_list = np.sort([bitstring_to_int(binary) for binary in bitstrings])
        count_dict = Counter(sorted_decimal_list)
        counts = [count_dict.get(i, 0) for i in range(2**len(samples[0]))]
        distribution = np.array(counts) / np.sum(counts)
        return distribution
    
    def _get_marginal_variation_dist(
        self,
        matrix: np.ndarray,
        bit_indices: np.ndarray, 
        ideal_marginal: np.ndarray
    ) -> np.ndarray:
        """Returns the variation distance between the ideal marginal and the
        empirical marginal. The empirical marginal is calculated from all the
        bitstrings up to the position specified by the bit_indices (including
        this position)."""
        row_index = bit_indices[0][0]
        k_order = len(bit_indices)
        column_inds = [bit_indices[j][1] for j in range(k_order)]
        submatrix = matrix[0 : row_index + 1, column_inds]
        empirical_distr = self._get_distribution_from_outcomes(submatrix)
        variation_distance = ideal_marginal - empirical_distr
        return variation_distance
    
    def _get_optimal_bitstring_in_decimal_for_first_column(
        self,
        S_matrix: np.ndarray,
        bit_indices: np.ndarray, 
        ideal_distrs: np.ndarray
    ) -> int:
        """Returns the optimal bitstring in decimal for the submatrix with index = 0."""
        dists: List = []
        for j in range(2**len(bit_indices)):
            S_matrix_copy = copy.deepcopy(S_matrix)
            bitstring = int_to_padded_bitstring(j, len(bit_indices))
            for i, bit in enumerate(bitstring):
                S_matrix_copy[bit_indices[i]] = bit
            variation_distance = 0.5*np.sum(np.abs(self._get_marginal_variation_dist(S_matrix_copy, bit_indices, ideal_distrs[0][1])))
            dists.append(variation_distance)
        optimal_ind = np.argmin(dists)
        return optimal_ind
    
    def _get_optimal_bitstring_in_decimal_for_column(
        self,
        S_matrix: np.ndarray,
        bit_indices: np.ndarray, 
        ideal_distrs: np.ndarray
    ) -> int:
        """Returns the optimal bitstring in decimal for a column with index > 0."""
        k_order = len(bit_indices)
        fixed_bits = tuple([S_matrix[bit_indices[i]] for i in range(k_order - 1)])
        possible_inds = [bitstring_to_int(fixed_bits + (0,)), bitstring_to_int(fixed_bits + (1,))]
        row_index = bit_indices[0][0]
        dists: List = []
        for j in possible_inds:
            S_matrix_copy = copy.deepcopy(S_matrix)
            bitstring = int_to_padded_bitstring(j, len(bit_indices))
            for i, bit in enumerate(bitstring):
                S_matrix_copy[bit_indices[i]] = bit
            column_inds = ideal_distrs[0][0]
            marginal = ideal_distrs[0][1]
            indices = [(row_index,) + (ind,) for ind in column_inds]
            variation_distance = 0.5*np.sum(np.abs(self._get_marginal_variation_dist(S_matrix_copy, indices, marginal)))
            for i in range(1, len(ideal_distrs)):
                column_inds = ideal_distrs[i][0]
                marginal = ideal_distrs[i][1]
                indices = [(row_index,) + (ind,) for ind in column_inds]
                variation_distance += 0.5*np.sum(np.abs(self._get_marginal_variation_dist(S_matrix_copy, indices, marginal)))
            dists.append(variation_distance)
        optimal_ind = possible_inds[np.argmin(dists)]
        return optimal_ind
    
    def _add_optimal_bitstring(
        self, 
        S_matrix: np.ndarray,
        bit_indices: np.ndarray, 
        ideal_distrs: np.ndarray,
        iteration_number: int
    ) -> None:
        """ Adds the bitstring to the S_matrix which minimizes the distance
        between the empirical and ideal distributions.Add bitstring where 
        the pointwise distance between the previous empirical distribution
        and the ideal distribution is the highest (where we need to add the
        highest amount of probability mass)."""
        if iteration_number == 0:
            optimal_ind = self._get_optimal_bitstring_in_decimal_for_first_column(S_matrix, bit_indices, ideal_distrs)
        else:
            optimal_ind = self._get_optimal_bitstring_in_decimal_for_column(S_matrix, bit_indices, ideal_distrs)
        bitstring = int_to_padded_bitstring(optimal_ind, len(bit_indices))
        for i, bit in enumerate(bitstring):
            S_matrix[bit_indices[i]] = bit

    def get_S_matrix(
        self, 
        n_modes: int, 
        n_rows: int, 
        k_order: int, 
        marginals: np.ndarray
    ) -> np.ndarray:
        """Takes an array of 1D discrete probability distributions
        which are the k-th order marginal distributions (e.g. of a GBS
        experiment) and approximates the full (GBS) distribution using
        Google's greedy algorithm.

        i) get_submatrix indices
        ii) add optimal bitstring until all rows of submatrix are filled
        iii) shuffle submatrix and increment iteration number

        The ground truth marginals are given in an array such that the jth
        element of that array contains a subarray with the marginals to be
        considered in that iteration, that is (k + j - 2) C (k) marginals,
        preceded by the indices of their corresponding columns
        e.g. for a k-th order approximation, the array containing the
        marginals has j subarrays. Each of these subarrays has two elements,
        a tuple with the column/mode indices and the marginal distribution.
        """
        assert (len(marginals) == n_modes - 1)
        for data in marginals:
            for d in data:
                marginal = d[1]
                assert (len(marginal) == 2**k_order)
                assert np.allclose(np.sum(marginal), 1, atol=1e-12)
        S_matrix = np.empty((n_rows, n_modes))
        for j in range(n_modes - 1):
            submatrix_inds = self._get_submatrix_indices(S_matrix.shape, k_order, j)
            for i in range(n_rows):
                self._add_optimal_bitstring(S_matrix, submatrix_inds[i], marginals[j], j)
            np.random.shuffle(S_matrix)
        return S_matrix

    def get_k_mode_marginal_distances_of_greedy_matrix(
        self, 
        S_matrix: np.ndarray, 
        k_order: int, 
        chain_marginals: np.ndarray
    ) -> np.ndarray:
        """Returns the variation distance of 'chained' k-mode marginals with respect
        to the given ideal marginals."""
        n_modes = S_matrix.shape[1]
        final_row_submatrix_indices = [self._get_submatrix_indices(S_matrix.shape, k_order, i)[-1] for i in range(n_modes - 1)]
        distances = [0.5*np.sum(np.abs(self._get_marginal_variation_dist(S_matrix, final_row_submatrix_indices[i], chain_marginals[i]))) for i in range(len(chain_marginals))]
        return np.array(distances)

    
