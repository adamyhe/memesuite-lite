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


@numba.njit(parallel=True, cache=True)
def _kmer_codes(X, X_lengths, w, alphabet_size):
	"""An internal function for encoding every length-`w` window as an integer.

	This function slides a width-`w` window across each sequence in a ragged
	(flat + cumulative-offsets) sequence array and encodes each window as a
	single base-`alphabet_size` integer (the same convention used everywhere
	else in this package for reverse-complementing: alphabet index `i` and
	`alphabet_size - 1 - i` are assumed to be complementary bases). Windows
	containing an unknown/ignored base (-1) are left as -1 in the output
	rather than being dropped, since dropping would require compacting the
	output inside a parallel loop; callers should filter with `codes >= 0`.

	This is a generic word-enumeration utility -- not specific to any one
	algorithm -- so it lives here rather than in a specific module.


	Parameters
	----------
	X: numpy.ndarray, shape=(-1,)
		A flat int8 array of alphabet indexes (-1 for unknown characters) for
		all sequences concatenated together. Sequences may have different
		lengths.

	X_lengths: numpy.ndarray, shape=(n_seqs+1,)
		The cumulative offsets demarcating each sequence's span within `X`.

	w: int
		The window width to encode.

	alphabet_size: int
		The number of characters in the alphabet.


	Returns
	-------
	codes: numpy.ndarray, shape=(total_windows,)
		The integer code of each potential window, in sequence order, or -1
		if that window contains an unknown character. `total_windows` is the
		sum over all sequences of `max(0, length - w + 1)`.

	owners: numpy.ndarray, shape=(total_windows,)
		The index of the sequence each entry of `codes` belongs to.
	"""

	n_seqs = len(X_lengths) - 1

	seq_lens = numpy.empty(n_seqs, dtype=numpy.int64)
	for s in range(n_seqs):
		seq_lens[s] = X_lengths[s + 1] - X_lengths[s]

	n_windows = numpy.zeros(n_seqs, dtype=numpy.int64)
	for s in range(n_seqs):
		n_windows[s] = max(seq_lens[s] - w + 1, 0)

	offsets = numpy.zeros(n_seqs + 1, dtype=numpy.int64)
	for s in range(n_seqs):
		offsets[s + 1] = offsets[s] + n_windows[s]

	total = offsets[n_seqs]
	codes = numpy.full(total, -1, dtype=numpy.int64)
	owners = numpy.empty(total, dtype=numpy.int64)

	for s in numba.prange(n_seqs):
		base = X_lengths[s]
		out0 = offsets[s]

		for i in range(n_windows[s]):
			owners[out0 + i] = s

			code = 0
			valid = True
			for j in range(w):
				idx = X[base + i + j]
				if idx < 0:
					valid = False
					break
				code = code * alphabet_size + idx

			if valid:
				codes[out0 + i] = code

	return codes, owners


def _reverse_complement_kmer_codes(codes, w, alphabet_size):
	"""Compute the reverse-complement code of each `_kmer_codes` code.

	Decodes each base-`alphabet_size` code into its `w` digits, reverses
	their order, and complements each digit as `alphabet_size - 1 - digit`
	(the same convention `_kmer_codes` and the rest of this package assume),
	then re-encodes. Fully vectorized over `codes`.


	Parameters
	----------
	codes: numpy.ndarray
		Non-negative integer codes from `_kmer_codes` (callers should filter
		out the -1 sentinel first).

	w: int
		The window width the codes were encoded with.

	alphabet_size: int
		The number of characters in the alphabet.


	Returns
	-------
	rc_codes: numpy.ndarray
		The reverse-complement code of each entry in `codes`.
	"""

	digits = numpy.empty((len(codes), w), dtype=numpy.int64)
	remaining = codes.copy()
	for j in range(w - 1, -1, -1):
		digits[:, j] = remaining % alphabet_size
		remaining //= alphabet_size

	complements = alphabet_size - 1 - digits

	rc_codes = numpy.zeros(len(codes), dtype=numpy.int64)
	for j in range(w):
		rc_codes = rc_codes * alphabet_size + complements[:, w - 1 - j]

	return rc_codes


@numba.njit(cache=True)
def _markov_transition_probs(X, X_lengths, order, alphabet_size, eps):
	"""An internal function for estimating an order-`m` Markov background.

	Pools all given (ragged) sequences and estimates, with pseudocount
	smoothing, the marginal (order-0) base frequency and, if `order` > 0, the
	conditional probability of each base given the `order` preceding bases.
	Used by `_markov_shuffle_sequences` to generate control sequences.


	Parameters
	----------
	X: numpy.ndarray, shape=(-1,)
		A flat int8 array of alphabet indexes (-1 for unknown characters) for
		all sequences concatenated together.

	X_lengths: numpy.ndarray, shape=(n_seqs+1,)
		The cumulative offsets demarcating each sequence's span within `X`.

	order: int
		The Markov order `m`. 0 means an i.i.d. background (no context).

	alphabet_size: int
		The number of characters in the alphabet.

	eps: float
		A pseudocount added to every count before normalizing, to avoid zero
		probabilities for contexts/bases never observed.


	Returns
	-------
	marginal: numpy.ndarray, shape=(alphabet_size,)
		The order-0 base frequency. Used as the model itself when `order` is
		0, and to seed the first `order` bases of each shuffled sequence
		(which have no preceding context yet) when `order` > 0.

	transitions: numpy.ndarray, shape=(alphabet_size**order, alphabet_size)
		Row `c` (a base-`alphabet_size`-encoded context of the `order`
		preceding bases, in the same left-to-right digit order as
		`_kmer_codes`) gives the conditional probability distribution of the
		next base. Unused (shape (1, alphabet_size)) when `order` is 0.
	"""

	n_seqs = len(X_lengths) - 1

	marginal_counts = numpy.full(alphabet_size, eps)
	n_contexts = alphabet_size ** order if order > 0 else 1
	trans_counts = numpy.full((n_contexts, alphabet_size), eps)

	for s in range(n_seqs):
		start, end = X_lengths[s], X_lengths[s + 1]

		for i in range(start, end):
			idx = X[i]
			if idx != -1:
				marginal_counts[idx] += 1

		if order > 0:
			for i in range(start + order, end):
				context = 0
				valid = True
				for j in range(order):
					c_idx = X[i - order + j]
					if c_idx == -1:
						valid = False
						break
					context = context * alphabet_size + c_idx

				idx = X[i]
				if valid and idx != -1:
					trans_counts[context, idx] += 1

	marginal = marginal_counts / marginal_counts.sum()

	transitions = numpy.empty((n_contexts, alphabet_size))
	for c in range(n_contexts):
		total = trans_counts[c].sum()
		for a in range(alphabet_size):
			transitions[c, a] = trans_counts[c, a] / total

	return marginal, transitions


@numba.njit(cache=True)
def _markov_shuffle_sequences(X_lengths, order, alphabet_size, marginal,
	transitions, seed):
	"""An internal function for sampling sequences from a Markov background.

	Generates one new sequence per entry of `X_lengths` (same length as the
	original), sampling each base from the order-`order` Markov chain defined
	by `marginal`/`transitions` (see `_markov_transition_probs`). The first
	`order` bases of each sequence have no preceding context yet, so they are
	sampled from the order-0 `marginal` distribution instead -- a standard,
	deliberately simple shortcut.

	Note this is a probabilistic *resampling* from the estimated Markov
	chain, not an exact letter-shuffle that preserves each individual
	sequence's own k-mer counts exactly (which the reference MEME suite's
	`fasta-shuffle-letters` does via a random Eulerian-circuit algorithm).
	This is a deliberate simplification: it is a standard, much simpler
	control-generation method that is good enough in practice, at the cost of
	not being an exact permutation of each sequence's own letters.


	Parameters
	----------
	X_lengths: numpy.ndarray, shape=(n_seqs+1,)
		The cumulative offsets of the sequences to match the length of. Only
		the lengths matter here, not the original content.

	order: int
		The Markov order `m` used to build `marginal`/`transitions`.

	alphabet_size: int
		The number of characters in the alphabet.

	marginal: numpy.ndarray, shape=(alphabet_size,)
		The order-0 base frequency, from `_markov_transition_probs`.

	transitions: numpy.ndarray, shape=(alphabet_size**order, alphabet_size)
		The order-`order` conditional base distribution, from
		`_markov_transition_probs`.

	seed: int
		A seed for numba's internal random number generator, for
		reproducibility.


	Returns
	-------
	X: numpy.ndarray, shape=(X_lengths[-1],)
		The flat int8 array of the newly sampled sequences, using the same
		cumulative offsets as `X_lengths`.
	"""

	numpy.random.seed(seed)

	n_seqs = len(X_lengths) - 1
	out = numpy.empty(X_lengths[n_seqs], dtype=numpy.int8)

	cum_marginal = numpy.cumsum(marginal)
	cum_marginal[alphabet_size - 1] = 1.0

	n_contexts = transitions.shape[0]
	cum_transitions = numpy.empty((n_contexts, alphabet_size))
	for c in range(n_contexts):
		acc = 0.0
		for a in range(alphabet_size):
			acc += transitions[c, a]
			cum_transitions[c, a] = acc
		cum_transitions[c, alphabet_size - 1] = 1.0

	for s in range(n_seqs):
		start, end = X_lengths[s], X_lengths[s + 1]

		context = 0
		for i in range(end - start):
			if order == 0 or i < order:
				u = numpy.random.random()
				base = alphabet_size - 1
				for a in range(alphabet_size):
					if u <= cum_marginal[a]:
						base = a
						break
			else:
				u = numpy.random.random()
				base = alphabet_size - 1
				for a in range(alphabet_size):
					if u <= cum_transitions[context, a]:
						base = a
						break

			out[start + i] = base

			if order > 0:
				context = (context * alphabet_size + base) % (alphabet_size ** order)

	return out


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
			raise ValueError(f"Character {char} in the alphabet " +
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