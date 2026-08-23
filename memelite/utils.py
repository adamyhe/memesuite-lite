# utils.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import numpy
import numba


@numba.njit('float64(float64, float64)', cache=True)
def logaddexp2(x, y):
	"""Calculate the logaddexp in a numerically stable manner in base 2.

	This function is a fast implementation of the logaddexp2 function that
	operates on two numbers and is numerically stable. It should mimic the
	functionality of numpy.logaddexp2 except that it does not have the overhead
	of working on numpy arrays.


	Parameters
	----------
	x: float32
		A single number in log space.

	y: float32
		Another single number in log space.


	Returns
	-------
	z: float32
		The result of log2(pow(2, x) + pow(2, y))
	"""

	if x == float("-inf") and y == float("-inf"):
		return float("-inf")

	if x == float("inf") or y == float("inf"):
		return float("inf")

	vmax, vmin = max(x, y), min(x, y)
	return vmax + math.log2(math.pow(2, vmin - vmax) + 1)


@numba.njit(cache=True)
def _pwm_to_mapping(log_pwm, bin_size):
	"""An internal method for calculating score <-> log p-value mappings.

	This function takes in a PWM consisting of log probabilities and outputs
	a mapping between observed scores (as a convolution of the PWM across a
	one-hot encoded sequence) and log p-values. This mapping is calculated
	quickly using dynamic programming scanning over all potential sequences.

	Importantly, the p-values are in log space meaning that values near zero
	at the start of the array are insignificant whereas those with large
	magnitude towards the end of the array are more statistically significant.


	Parameters
	----------
	log_pwm: numpy.ndarray, shape=(len(alphabet), length)
		A position-weight matrix containing a motif encoded as the log
		probability of any character in any position.

	bin_size: float
		The size of the score bins to map to p-values. The smaller this value,
		the more bins, indicating higher precision but also longer calculation
		time.


	Returns
	-------
	smallest: int
		The number of bins between true zero and the smallest value in the
		array. In other words, the offset to subtract from binned scores to get
		p-values.

	log1mcdf: numpy.ndarray
		The log of 1 minus the cdf, or in other words, the log p-values
		associated with each score bin.
	"""

	n, l = log_pwm.shape

	log_bg = math.log2(0.25)
	int_log_pwm = numpy.round(log_pwm / bin_size).astype(numpy.int32)

	smallest, largest = 9999999, -9999999
	log_pwm_min_csum, log_pwm_max_csum = 0, 0
	for i in range(l):
		log_pwm_min = 9999999
		log_pwm_max = -9999999

		for j in range(n):
			log_pwm_min = min(log_pwm_min, int_log_pwm[j, i])
			log_pwm_max = max(log_pwm_max, int_log_pwm[j, i])

		log_pwm_min_csum += log_pwm_min
		log_pwm_max_csum += log_pwm_max

		smallest = min(smallest, log_pwm_min_csum)
		largest = max(largest, log_pwm_max_csum)

	largest += l

	logpdf = numpy.empty(largest - smallest + 1)
	old_logpdf = -numpy.inf * numpy.ones(largest - smallest + 1)
	for i in range(n):
		idx = int_log_pwm[i, 0] - smallest
		old_logpdf[idx] = logaddexp2(old_logpdf[idx], log_bg)

	for i in range(1, l):
		for j in range(largest - smallest + 1):
			logpdf[j] = -numpy.inf

		for j, x in enumerate(old_logpdf):
			if x != -numpy.inf:
				for k in range(n):
					idx = j + int_log_pwm[k, i]
					logpdf[idx] = logaddexp2(logpdf[idx], log_bg + x)

		for j in range(largest - smallest + 1):
			old_logpdf[j] = logpdf[j]

	for i in range(len(logpdf) - 2, -1, -1):
		logpdf[i] = logaddexp2(logpdf[i], logpdf[i + 1])

	return smallest, logpdf


@numba.njit(parallel=True, cache=True)
def _all_pwm_to_mapping(motifs, motif_lengths, bin_size):
	n = len(motif_lengths) - 1

	smallests = numpy.empty(n, dtype='int64')
	logpdfs = [numpy.empty(0) for i in range(n)]

	for i in numba.prange(n):
		s, e = motif_lengths[i], motif_lengths[i+1]

		smallest, logpdf = _pwm_to_mapping(motifs[:, s:e], bin_size)
		smallests[i] = smallest
		logpdfs[i] = logpdf

	return smallests, logpdfs


def _pvalue_score_thresholds(pwms_concat, lengths, bin_size, threshold):
	"""Convert a p-value threshold into a per-motif raw score threshold.

	This wraps `_all_pwm_to_mapping`'s score-to-p-value mapping, inverting it
	to find, for each motif, the smallest raw score whose p-value is below
	`threshold`. If no achievable score reaches `threshold`, that motif's
	threshold is set to infinity (nothing can ever qualify).


	Parameters
	----------
	pwms_concat: numpy.ndarray, shape=(len(alphabet), total_width)
		The concatenated log-odds PWMs to threshold.

	lengths: numpy.ndarray
		The cumulative offsets demarcating each PWM's span within
		`pwms_concat`. Has `len(pwms_concat's motifs) + 1` entries.

	bin_size: float
		The size of the bins discretizing the PWM scores, as in
		`_all_pwm_to_mapping`.

	threshold: float
		The p-value threshold to convert.


	Returns
	-------
	score_thresholds: numpy.ndarray, shape=(len(lengths) - 1,)
		The raw score threshold for each motif.

	smallest: numpy.ndarray
		The per-motif bin offsets from `_all_pwm_to_mapping`, in case the
		caller also needs to convert an arbitrary observed score back into a
		p-value (e.g. to report which threshold an optimizing search chose).

	score_to_pvals: list of numpy.ndarray
		The per-motif score-bin-to-log2(p-value) mappings from
		`_all_pwm_to_mapping`, for the same reason.
	"""

	log_threshold = math.log2(threshold)
	smallest, score_to_pvals = _all_pwm_to_mapping(pwms_concat,
		lengths.astype(numpy.uint64), bin_size)

	n = len(lengths) - 1
	score_thresholds = numpy.empty(n, dtype=numpy.float64)
	for i in range(n):
		idx = numpy.where(score_to_pvals[i] < log_threshold)[0]
		if len(idx) > 0:
			score_thresholds[i] = (idx[0] + smallest[i]) * bin_size
		else:
			score_thresholds[i] = float("inf")

	return score_thresholds, smallest, score_to_pvals


def characters(pwm, alphabet=['A', 'C', 'G', 'T'], force=False, allow_N=False):
	"""Converts a PWM/one-hot encoding to a string sequence.

	This function takes in a PWM or one-hot encoding and converts it to the
	most likely sequence. When the input is a one-hot encoding, this is the
	opposite of the `one_hot_encoding` function.


	Parameters
	----------
	pwm: numpy.array, shape=(len(alphabet), seq_len)
		A numeric representation of the sequence. This can be one-hot encoded
		or contain numeric values. These numerics can be probabilities but can
		also be frequencies.

	alphabet : set or tuple or list
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor. This is used to determine the
		letters in the returned sequence. Default is the DNA alphabet.

	force: bool, optional
		Whether to force a sequence to be produced even when there are ties.
		At each position that there is a tight, the character earlier in the
		sequence will be used. Default is False.
  
	allow_N: bool, optional
		Whether to allow the return of the character 'N' in the sequence, i.e.
		if pwm at a position is all 0's return N. Default is False.


	Returns
	-------
	seq: str
		A string where the length is the second dimension of PWM.
	"""
 
	if len(pwm.shape) == 3 and pwm.shape[0] == 1:
		pwm = pwm[0]

	if len(pwm.shape) != 2:
		raise ValueError("PWM must have two dimensions where the " +
			"first dimension is the length of the alphabet and the second " +
			"dimension is the length of the sequence.")

	if pwm.shape[0] != len(alphabet):
		raise ValueError("PWM must have the same alphabet size as the " +
			"provided alphabet.")

	pwm_ismax = pwm == pwm.max(axis=0, keepdims=True)
	if pwm_ismax.sum(axis=0).max() > 1 and force == False and allow_N == False:
		raise ValueError("At least one position in the PWM has multiple " +
			"letters with the same probability.")

	alphabet = numpy.array(alphabet)
	if not isinstance(pwm, numpy.ndarray):
		pwm = pwm.numpy(force=True)

	if allow_N:
		n_inds = numpy.where(pwm.sum(axis=0) == 0)[0]
		dna_chars = alphabet[pwm.argmax(axis=0)]
		dna_chars[n_inds] = 'N'
	else:
		dna_chars = alphabet[pwm.argmax(axis=0)]
	
	return ''.join(dna_chars)


@numba.njit("void(int8[:, :], int8[:], int8[:])", cache=True)
def _fast_one_hot_encode(X_ohe, seq, mapping):
	"""An internal function for quickly converting bytes to one-hot indexes."""

	for i in range(len(seq)):
		idx = mapping[seq[i]]
		if idx == -1:
			continue

		if idx == -2:
			raise ValueError("Encountered character that is not in " + 
				"`alphabet` or in `ignore`.")
			
		X_ohe[i, idx] = 1


def one_hot_encode(sequence, alphabet=['A', 'C', 'G', 'T'], dtype=numpy.int8, 
	ignore=['N'], desc=None, verbose=False, **kwargs):
	"""Converts a string or list of characters into a one-hot encoding.

	This function will take in a string and convert it into a one-hot
	encoding. Each character is assumed to be a different symbol, e.g. 'ACGT'
	is assumed to be a sequence of four characters that are matched against
	the alphabet.

	Although this function will be used here primarily to convert nucleotide
	sequences into one-hot encoding with an alphabet of size 4, in principle
	this function can be used for any types of sequences as long as each
	symbol is a single character.

	Parameters
	----------
	sequence : str
		The sequence to convert to a one-hot encoding. Must be a string where
		each character is a single symbol in the alphabet.

	alphabet : set or tuple or list
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. Default is ['A', 'C', 'G', 'T'].

	dtype : str or torch.dtype, optional
		The data type of the returned encoding. Default is int8.

	ignore: list, optional
		A list of characters to ignore in the sequence, meaning that no bits
		are set to 1 in the returned one-hot encoding. Put another way, the
		sum across characters is equal to 1 for all positions except those
		where the original sequence is in this list. Default is ['N'].


	Returns
	-------
	ohe : numpy.ndarray
		A binary matrix of shape (alphabet_size, sequence_length) where
		alphabet_size is the number of unique elements in the sequence and
		sequence_length is the length of the input sequence.
	"""

	for char in ignore:
		if char in alphabet:
			raise ValueError("Character {} in the alphabet ".format(char) + 
				"and also in the list of ignored characters.")

	if isinstance(alphabet, list):
		alphabet = ''.join(alphabet)

	ignore = ''.join(ignore)

	e = "utf8"
	seq_idxs = numpy.frombuffer(bytearray(sequence, e), dtype=numpy.int8)
	alpha_idxs = numpy.frombuffer(bytearray(alphabet, e), dtype=numpy.int8)
	ignore_idxs = numpy.frombuffer(bytearray(ignore, e), dtype=numpy.int8)

	one_hot_mapping = numpy.zeros(256, dtype=numpy.int8) - 2
	for i, idx in enumerate(alpha_idxs):
		one_hot_mapping[idx] = i

	for i, idx in enumerate(ignore_idxs):
		one_hot_mapping[idx] = -1

	n, m = len(sequence), len(alphabet)

	one_hot_encoding = numpy.zeros((n, m), dtype=numpy.int8)
	_fast_one_hot_encode(one_hot_encoding, seq_idxs, one_hot_mapping)
	return one_hot_encoding.astype(dtype).T