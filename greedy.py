from typing import Tuple
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


    def _get_marginal_distribution_from_outcomes(self, marginals, samples: np.ndarray) -> np.ndarray:
        '''Turns list of outcomes (bitstrings) into empiral marginal distributions of input marginals'''
        reduced_outcomes = samples[:,marginals]
        return self._get_distribution_from_outcomes(reduced_outcomes)
        
    def _get_marginal_variation_dist(
        self,
        matrix: np.ndarray,
        bit_indices: np.ndarray, 
        ideal_marginal: np.ndarray
    ) -> np.ndarray:
        """Returns the variation distance between the ideal marginal and the
        empirical marginal at the position specified by the bit_indices input
        variable in the input matrix."""
        row_index = bit_indices[0][0]
        assert (row_index > 0)
        k_order = len(bit_indices)
        column_inds = [bit_indices[j][1] for j in range(k_order)]
        submatrix = matrix[0 : row_index, column_inds]
        empirical_distr = self._get_distribution_from_outcomes(submatrix)
        variation_distance = ideal_marginal - empirical_distr
        return variation_distance
    
    def _get_optimal_bitstring_in_decimal_for_first_column(
        self,
        S_matrix: np.ndarray,
        bit_indices: np.ndarray, 
        ideal_distr: np.ndarray
    ) -> int:
        """Returns the optimal bitstring in decimal for the column with index = 0."""
        row_index = bit_indices[0][0]
        if row_index == 0:
            max_ind = np.argmax(ideal_distr)
        else:
            variation_distance = self._get_marginal_variation_dist(S_matrix, bit_indices, ideal_distr)
            max_ind = np.argmax(variation_distance)
        return max_ind
    
    def _get_optimal_bitstring_in_decimal_for_column(
        self,
        S_matrix: np.ndarray,
        bit_indices: np.ndarray, 
        ideal_distr: np.ndarray
    ) -> int:
        """Returns the optimal bitstring in decimal for a column with index > 0."""
        row_index = bit_indices[0][0]
        k_order = len(bit_indices)
        fixed_bits = tuple([S_matrix[bit_indices[i]] for i in range(k_order - 1)])
        possible_inds = [bitstring_to_int(fixed_bits + (0,)), bitstring_to_int(fixed_bits + (1,))]
        if row_index == 0:
            max_ind = possible_inds[np.argmax(ideal_distr[possible_inds[0]:possible_inds[1] + 1])]
        else:
            variation_distance = self._get_marginal_variation_dist(S_matrix, bit_indices, ideal_distr)
            max_ind = possible_inds[np.argmax(variation_distance[possible_inds[0]:possible_inds[1] + 1])]
        return max_ind
    
    def _add_optimal_bitstring(
        self, 
        S_matrix: np.ndarray,
        bit_indices: np.ndarray, 
        ideal_distr: np.ndarray,
        iteration_number: int
    ) -> None:
        """ Adds the bitstring to the S_matrix which minimizes the distance
        between the empirical and ideal distributions.Add bitstring where 
        the pointwise distance between the previous empirical distribution
        and the ideal distribution is the highest (where we need to add the
        highest amount of probability mass)."""
        if iteration_number == 0:
            max_ind = self._get_optimal_bitstring_in_decimal_for_first_column(S_matrix, bit_indices, ideal_distr)
        else:
            max_ind = self._get_optimal_bitstring_in_decimal_for_column(S_matrix, bit_indices, ideal_distr)
        bitstring = int_to_padded_bitstring(max_ind, len(bit_indices))
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
        """
        assert (len(marginals) == n_modes - 1)
        for marginal in marginals:
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
        marginals: np.ndarray
    ) -> np.ndarray:
        """Returns the variation distance of 'chained' k-mode marginals with respect
        to the given ideal marginals."""
        final_row_submatrix_indices = [self._get_submatrix_indices(S_matrix.shape, k_order, i)[-1] for i in range(len(marginals))]
        distances = [0.5*np.sum(np.abs(self._get_marginal_variation_dist(S_matrix, final_row_submatrix_indices[i], marginals[i]))) for i in range(len(marginals))]
        return np.array(distances)
