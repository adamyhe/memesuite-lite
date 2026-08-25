# streme.py
# Author: Adam He <adamyhe@gmail.com>

import math
import numba
import numpy
import pandas
import scipy.stats

from .io import _load_ragged_sequences
from .utils import _kmer_codes
from .utils import _reverse_complement_kmer_codes
from .utils import _markov_transition_probs
from .utils import _markov_shuffle_sequences
from .utils import characters


@numba.njit(parallel=True, fastmath=True, cache=True)
def _streme_best_sites(X, X_lengths, pwm_fwd, pwm_rc, w, reverse_complement):
	"""An internal function for finding each sequence's best-scoring site.

	This is the core scanning loop for STREME's seed refinement, and is a
	ragged-sequence (variable length) analogue of `centrimo`'s
	`_centrimo_best_sites`: for each sequence, finds the single best-scoring
	site for one motif (optionally combining forward and reverse complement
	strands by taking whichever scores higher) and records its position,
	winning strand, and raw score. Unlike `centrimo`, ties are broken by
	keeping the first (leftmost) position, preferring the forward strand on
	an exact tie -- the same deterministic tie-break `spamo` uses, for the
	same reason (there is no meaningful "average" of two different site
	positions).


	Parameters
	----------
	X: numpy.ndarray, shape=(-1,)
		A flat int8 array of alphabet indexes (-1 for unknown characters) for
		all sequences concatenated together. Sequences may have different
		lengths.

	X_lengths: numpy.ndarray, shape=(n_seqs+1,)
		The cumulative offsets demarcating each sequence's span within `X`.

	pwm_fwd: numpy.ndarray, shape=(len(alphabet), w)
		The motif's log-odds PWM.

	pwm_rc: numpy.ndarray, shape=(len(alphabet), w)
		The reverse complement of `pwm_fwd`. Read even when
		`reverse_complement` is False, so the caller must still supply it.

	w: int
		The width of the motif.

	reverse_complement: bool
		Whether to also score the reverse complement strand at each position
		and keep whichever strand scores higher.


	Returns
	-------
	scores: numpy.ndarray, shape=(n_seqs,)
		The raw score of each sequence's best-scoring site. Sequences too
		short for a single site of width `w` get a large negative sentinel.

	positions: numpy.ndarray, shape=(n_seqs,), dtype=int32
		The start position of each sequence's best site, or -1 if the
		sequence is too short for width `w`.

	strands: numpy.ndarray, shape=(n_seqs,), dtype=int8
		0 if the winning site was on the forward strand, 1 if reverse
		complement. Meaningless where `positions == -1`.
	"""

	n_seqs = len(X_lengths) - 1
	scores = numpy.empty(n_seqs, dtype=numpy.float64)
	positions = numpy.full(n_seqs, -1, dtype=numpy.int32)
	strands = numpy.zeros(n_seqs, dtype=numpy.int8)

	for s in numba.prange(n_seqs):
		start = X_lengths[s]
		n_valid = X_lengths[s + 1] - start - w + 1

		best_score = -1e300
		best_i = -1
		best_strand = 0

		for i in range(n_valid):
			# A window overlapping any erased position (-2, distinct from
			# the ordinary -1 "unknown base" sentinel) is disqualified
			# entirely, not merely skipped at that one position -- treating
			# an erased position as a neutral (contributes-nothing) skip,
			# the way -1 is treated, would make heavily-erased windows
			# score suspiciously close to 0, which is *higher* than a real
			# background window's typically-negative score, making erased
			# regions look artificially attractive and get "rediscovered"
			# every round. See `_streme_erase`.
			erased = False
			for j in range(w):
				if X[start + i + j] == -2:
					erased = True
					break

			if erased:
				continue

			score = 0.0
			for j in range(w):
				idx = X[start + i + j]
				if idx == -1:
					continue
				score += pwm_fwd[idx, j]

			if score > best_score:
				best_score = score
				best_i = i
				best_strand = 0

			if reverse_complement:
				score_rc = 0.0
				for j in range(w):
					idx = X[start + i + j]
					if idx == -1:
						continue
					score_rc += pwm_rc[idx, j]

				if score_rc > best_score:
					best_score = score_rc
					best_i = i
					best_strand = 1

		scores[s] = best_score
		positions[s] = best_i
		strands[s] = best_strand

	return scores, positions, strands


@numba.njit(parallel=True, cache=True)
def _streme_erase(X, X_lengths, scores, positions, strands, pwm_fwd, pwm_rc,
	w, threshold):
	"""An internal function for erasing a motif's sites from sequences.

	For each sequence whose best site (from `_streme_best_sites`, on the
	same motif being erased) scores at or above `threshold`, erases (sets to
	-2) only the positions within that site whose log-odds contribution on
	the winning strand is positive. If no position is positive (which can
	happen with a very degenerate PWM), erases just the middle position
	instead, so that some progress is always made and the same word cannot
	be found again unchanged. This mirrors the real STREME binary's erase
	step exactly.

	-2 is deliberately distinct from the -1 sentinel used elsewhere in this
	package for an ordinary unknown/"N" base: `_streme_best_sites` and
	`_kmer_codes` treat any window *containing* a -2 as entirely
	disqualified, not merely skipped at that one position. Treating erased
	positions the same as -1 (silently skipped, contributing nothing to the
	score) would make heavily-erased windows score close to 0 -- higher than
	a real background window's typically-negative score -- so erased
	regions would look artificially attractive and get "rediscovered" every
	subsequent round instead of actually being removed from consideration.

	Mutates `X` in place. Safe to run with `numba.prange` because each
	sequence's erasable span is disjoint from every other sequence's.


	Parameters
	----------
	X: numpy.ndarray, shape=(-1,)
		A flat int8 array of alphabet indexes, mutated in place.

	X_lengths: numpy.ndarray, shape=(n_seqs+1,)
		The cumulative offsets demarcating each sequence's span within `X`.

	scores, positions, strands: numpy.ndarray
		This motif's best-site outputs from `_streme_best_sites` on `X`.

	pwm_fwd, pwm_rc: numpy.ndarray, shape=(len(alphabet), w)
		The motif's log-odds PWM and its reverse complement.

	w: int
		The width of the motif.

	threshold: float
		The raw score a site must reach to be erased.
	"""

	n_seqs = len(X_lengths) - 1

	for s in numba.prange(n_seqs):
		if positions[s] < 0 or scores[s] < threshold:
			continue

		start = X_lengths[s] + positions[s]
		strand = strands[s]

		any_positive = False
		for j in range(w):
			idx = X[start + j]
			if idx == -1:
				continue

			contrib = pwm_fwd[idx, j] if strand == 0 else pwm_rc[idx, j]
			if contrib > 0:
				X[start + j] = -2
				any_positive = True

		if not any_positive:
			X[start + w // 2] = -2


def _code_to_pwm(code, w, alphabet_size, eps):
	"""Build an initial probability matrix from a `_kmer_codes` seed word.

	Decodes `code` back into its `w` bases and places (1 - (alphabet_size-1)
	* eps) at each position's seed base and `eps` elsewhere, so every column
	sums to exactly 1.
	"""

	pwm = numpy.full((alphabet_size, w), eps)
	remaining = code

	for j in range(w - 1, -1, -1):
		base = remaining % alphabet_size
		remaining //= alphabet_size
		pwm[base, j] = 1.0 - (alphabet_size - 1) * eps

	return pwm


def _reestimate_pwm(X, X_lengths, scores, positions, strands, w,
	alphabet_size, threshold, eps):
	"""Re-estimate a probability matrix from the current round's best sites.

	Builds a position frequency matrix (with `eps` pseudocounts) from every
	sequence's best site that scores at or above `threshold`, orienting each
	site to the forward strand (reverse-complementing it first if the
	reverse complement strand won) before counting. This is small
	(O(n_seqs * w)) and not the throughput bottleneck, so it is plain
	Python/NumPy rather than a numba kernel -- doing this reduction inside a
	`numba.prange` loop would require multiple threads to accumulate into the
	same shared (alphabet_size, w) array, which is not race-free.
	"""

	pwm = numpy.full((alphabet_size, w), eps)

	for s in range(len(X_lengths) - 1):
		if positions[s] < 0 or scores[s] < threshold:
			continue

		start = X_lengths[s] + positions[s]
		window = X[start:start + w]

		if strands[s] == 1:
			window = window[::-1]
			window = numpy.where(window < 0, window, alphabet_size - 1 - window)

		for j in range(w):
			idx = window[j]
			if idx >= 0:
				pwm[idx, j] += 1

	pwm /= pwm.sum(axis=0, keepdims=True)
	return pwm


def _enrichment_pvalues(pos_counts, neg_counts, n_pos, n_neg, use_fisher,
	bernoulli):
	"""Score primary-vs-control enrichment of one or more presence counts.

	Shared by `_streme_seed_candidates` (scoring many candidate words at
	once) and `_best_threshold` (scoring many candidate score cutoffs at
	once): both reduce to "how many of `n_pos` primary sequences vs. `n_neg`
	control sequences contain something," tested exactly as the real STREME
	binary does (verified against its source, `streme-utils.c`/`.h`).

	Fisher's exact test (one-sided, "greater") is computed via the
	`hypergeom.sf` identity rather than `scipy.stats.fisher_exact`, since the
	latter cannot be vectorized over arrays of counts.


	Parameters
	----------
	pos_counts, neg_counts: numpy.ndarray
		The number of primary/control sequences containing each candidate
		(a word, or a site scoring above a candidate threshold).

	n_pos, n_neg: int
		The total number of primary/control sequences.

	use_fisher: bool
		Whether to use Fisher's exact test (via `hypergeom.sf`) or the
		Binomial test. The real STREME binary switches to the Binomial test
		when the primary and control average sequence lengths differ by at
		least 0.01%, and uses Fisher's exact test otherwise.

	bernoulli: float or None
		The Binomial test's null probability (ignored when `use_fisher`).


	Returns
	-------
	p_values: numpy.ndarray
		The one-sided enrichment p-value for each entry of `pos_counts`.
	"""

	if use_fisher:
		return scipy.stats.hypergeom.sf(pos_counts - 1, n_pos + n_neg,
			pos_counts + neg_counts, n_pos)

	return scipy.stats.binom.sf(pos_counts - 1, pos_counts + neg_counts,
		bernoulli)


def _binomial_bernoulli(n_pos, n_neg, avg_pos_len, avg_neg_len, avg_w):
	"""The Binomial test's null probability, matching the real STREME binary.

	`pos_sites`/`neg_sites` approximate the total number of candidate sites
	across all primary/control sequences (each sequence contributing
	`avg_len - avg_w + 1` candidate positions, floored at 1). `avg_w` is a
	single value fixed across all candidate widths (confirmed from source,
	`streme-utils.c`), not the width of whichever candidate is being scored.
	"""

	pos_sites = n_pos * max(1.0, avg_pos_len - avg_w + 1)
	neg_sites = n_neg * max(1.0, avg_neg_len - avg_w + 1)
	return pos_sites / (pos_sites + neg_sites)


def _streme_seed_candidates(X_pos, Xl_pos, X_neg, Xl_neg, w, alphabet_size,
	nref, n_pos, n_neg, use_fisher, bernoulli):
	"""Score every distinct word of width `w` and return the top `nref`.

	This replaces the real STREME binary's generalized-suffix-tree seed
	search with exact, vectorized k-mer counting: every length-`w` window of
	every sequence is encoded as an integer (`_kmer_codes`), forward and
	reverse-complement occurrences of the same site are canonicalized to one
	word (`min(code, rc_code)`, matching the source's own forward/RC seed
	deduplication), and each observed word's exact enrichment p-value is
	computed directly and vectorized (`_enrichment_pvalues`) rather than via
	the source's approximate two-stage (top-25-then-exact-count) narrowing --
	since our count is already exact, that intermediate stage buys nothing.


	Returns
	-------
	codes: numpy.ndarray
		The canonical word codes of the `nref` most enriched words (fewer if
		fewer than `nref` distinct words were observed at all).
	"""

	pos_codes, pos_owners = _kmer_codes(X_pos, Xl_pos, w, alphabet_size)
	neg_codes, neg_owners = _kmer_codes(X_neg, Xl_neg, w, alphabet_size)

	pos_mask, neg_mask = pos_codes >= 0, neg_codes >= 0
	pos_codes, pos_owners = pos_codes[pos_mask], pos_owners[pos_mask]
	neg_codes, neg_owners = neg_codes[neg_mask], neg_owners[neg_mask]

	if len(pos_codes) == 0:
		return numpy.array([], dtype=numpy.int64)

	pos_canon = numpy.minimum(pos_codes,
		_reverse_complement_kmer_codes(pos_codes, w, alphabet_size))

	if len(neg_codes) > 0:
		neg_canon = numpy.minimum(neg_codes,
			_reverse_complement_kmer_codes(neg_codes, w, alphabet_size))
	else:
		neg_canon = neg_codes

	pos_pairs = numpy.unique(numpy.stack([pos_owners, pos_canon], axis=1), axis=0)
	pos_words, pos_word_counts = numpy.unique(pos_pairs[:, 1], return_counts=True)

	if len(neg_canon) > 0:
		neg_pairs = numpy.unique(numpy.stack([neg_owners, neg_canon], axis=1), axis=0)
		neg_words, neg_word_counts = numpy.unique(neg_pairs[:, 1], return_counts=True)
	else:
		neg_words = numpy.array([], dtype=numpy.int64)
		neg_word_counts = numpy.array([], dtype=numpy.int64)

	all_words = numpy.union1d(pos_words, neg_words)
	pos_aligned = numpy.zeros(len(all_words), dtype=numpy.int64)
	neg_aligned = numpy.zeros(len(all_words), dtype=numpy.int64)
	pos_aligned[numpy.searchsorted(all_words, pos_words)] = pos_word_counts
	neg_aligned[numpy.searchsorted(all_words, neg_words)] = neg_word_counts

	p_values = _enrichment_pvalues(pos_aligned, neg_aligned, n_pos, n_neg,
		use_fisher, bernoulli)

	rank = numpy.argsort(p_values)[:nref]
	return all_words[rank]


def _best_threshold(pos_scores, neg_scores, n_pos, n_neg, use_fisher,
	bernoulli):
	"""Find the score cutoff maximizing primary-vs-control enrichment.

	Sweeps every distinct score achieved by any primary or control
	sequence's best site as a candidate cutoff, vectorized via sort +
	`searchsorted` + `_enrichment_pvalues` -- the same pattern `centrimo`'s
	`optimize_score` branch already uses for its own threshold search.
	"""

	grid = numpy.unique(numpy.concatenate([pos_scores, neg_scores]))
	sorted_pos, sorted_neg = numpy.sort(pos_scores), numpy.sort(neg_scores)

	pos_counts = len(sorted_pos) - numpy.searchsorted(sorted_pos, grid, side='left')
	neg_counts = len(sorted_neg) - numpy.searchsorted(sorted_neg, grid, side='left')

	p_values = _enrichment_pvalues(pos_counts, neg_counts, n_pos, n_neg,
		use_fisher, bernoulli)

	best = int(numpy.argmin(p_values))
	return (float(grid[best]), int(pos_counts[best]), int(neg_counts[best]),
		float(p_values[best]))


def _streme_refine(X_pos, Xl_pos, X_neg, Xl_neg, code, w, alphabet_size, eps,
	niter, n_pos, n_neg, use_fisher, bernoulli, reverse_complement):
	"""Refine one seed word into a PWM via iterative threshold search + EM.

	Each iteration: score every sequence's best site under the current PWM
	(`_streme_best_sites`), find the cutoff maximizing primary-vs-control
	enrichment (`_best_threshold`), and re-estimate the PWM from primary
	sites scoring above that cutoff (`_reestimate_pwm`). Stops as soon as a
	round fails to improve on the best p-value seen so far, and returns the
	state from that best round (not necessarily the last one run).
	"""

	pwm = _code_to_pwm(code, w, alphabet_size, eps)
	best = None

	for _ in range(niter):
		log_pwm_fwd = numpy.log2(pwm + eps) - math.log2(1.0 / alphabet_size)
		log_pwm_rc = log_pwm_fwd[::-1, ::-1].copy()

		pos_scores, pos_pos, pos_strand = _streme_best_sites(X_pos, Xl_pos,
			log_pwm_fwd, log_pwm_rc, w, reverse_complement)
		neg_scores, neg_pos, neg_strand = _streme_best_sites(X_neg, Xl_neg,
			log_pwm_fwd, log_pwm_rc, w, reverse_complement)

		threshold, pos_count, neg_count, p_value = _best_threshold(pos_scores,
			neg_scores, n_pos, n_neg, use_fisher, bernoulli)

		if best is not None and p_value >= best['p_value']:
			break

		best = {
			'pwm': pwm, 'log_pwm_fwd': log_pwm_fwd, 'log_pwm_rc': log_pwm_rc,
			'threshold': threshold, 'pos_count': pos_count,
			'neg_count': neg_count, 'p_value': p_value,
			'pos_scores': pos_scores, 'pos_pos': pos_pos, 'pos_strand': pos_strand,
			'neg_scores': neg_scores, 'neg_pos': neg_pos, 'neg_strand': neg_strand,
		}

		pwm = _reestimate_pwm(X_pos, Xl_pos, pos_scores, pos_pos, pos_strand,
			w, alphabet_size, threshold, eps)

	return best


def _subset_ragged(X, X_lengths, indices):
	"""Extract a subset of sequences from a ragged array as fresh copies.

	Unlike a plain slice, `numpy.concatenate` always allocates a new array,
	so the result is safe to mutate (e.g. via `_streme_erase`) without
	affecting `X`.
	"""

	if len(indices) == 0:
		return numpy.array([], dtype=X.dtype), numpy.array([0], dtype=numpy.int64)

	pieces = [X[X_lengths[i]:X_lengths[i + 1]] for i in indices]
	lengths = numpy.cumsum([0] + [len(p) for p in pieces]).astype(numpy.int64)
	return numpy.concatenate(pieces), lengths


def streme(sequences, control_sequences=None, alphabet=('A', 'C', 'G', 'T'),
	minw=8, maxw=15, order=2, nmotifs=None, threshold=0.05, use_evalue=False,
	patience=3, hofract=0.1, nref=4, niter=20, reverse_complement=True,
	eps=0.0001, random_state=None, return_site_positions=False, n_jobs=-1):
	"""An implementation of the STREME algorithm from the MEME suite.

	This function implements "Simple, Thorough, Rapid, Enriched Motif
	Elicitation" (STREME), the MEME suite's de novo motif discovery
	algorithm. Unlike `fimo`/`centrimo`/`spamo` (which all take existing
	motifs as input), STREME *discovers* novel motifs directly from a set of
	sequences, by finding short words that are more common in the given
	("primary") sequences than in a set of control sequences, refining each
	into a PWM, and repeating with the found sites erased until no further
	motif is significant. This is useful for characterizing motifs without
	needing a pre-existing PWM, e.g. to validate a motif surfaced by an
	attribution method on a machine learning model.

	Algorithm (verified against Bailey 2021, *Bioinformatics* 37(18):2834-40,
	the official MEME suite documentation, and the real MEME suite v5.5.9 C
	source, `src/streme.c`/`src/streme-utils.c`): the primary sequences (and
	control sequences, whether given or generated -- see `order` below) are
	each split once into a search fraction and a held-out fraction
	(`hofract`). Each round: every width in `[minw, maxw]` is searched and
	refined independently (candidate seed words are scored by primary-vs-
	control enrichment, the top `nref` per width are each refined into a PWM
	via iterative threshold search + re-estimation -- see `_streme_refine` --
	and the single most enriched refined candidate across all widths is this
	round's motif). Its significance is then assessed on the *held-out*
	fraction only, at the threshold and PWM already fixed during search, so
	the search process cannot inflate its own reported significance. If
	significant (`threshold`, on the p-value by default or the E-value if
	`use_evalue` is True), it is added to the output; regardless, its sites
	are erased (masked, so future rounds cannot simply rediscover the same
	site) from every sequence set before the next round. Search stops after
	`patience` consecutive non-significant rounds, or once `nmotifs` motifs
	have been found if `nmotifs` is given (which also disables the
	significance requirement, reporting exactly `nmotifs` motifs regardless).

	Three implementation choices deliberately diverge from the reference
	STREME binary, each because reproducing it exactly would require a
	substantial independent piece of engineering with no accuracy benefit for
	realistic inputs on this package's numpy/numba-based architecture:

	1. Seed search: the reference binary builds a generalized suffix tree
	   with approximate p-value pruning. This implementation instead encodes
	   every candidate word as an integer and counts its exact primary/
	   control presence directly (see `_streme_seed_candidates`) -- the exact
	   same statistic, computed exactly rather than approximately, at the
	   cost of the suffix tree's better asymptotic scaling for very large
	   inputs.

	2. Scoring background: the reference binary scores PWMs against an
	   `order`-th order Markov background during refinement (with `order`
	   also controlling shuffle-based control generation), but a plain
	   0-order (uniform) background specifically for the erase step,
	   regardless of search order. Every other kernel in this package
	   (`fimo`, `centrimo`, `spamo`) already assumes a fixed uniform 0.25
	   background for PWM scoring -- there is no context-conditioned scoring
	   convention anywhere in this codebase to build on. Here, `order`
	   therefore governs *only* automatic control generation (see below);
	   refinement and erasing both use the same uniform-background
	   convention as every other kernel in this package, which conveniently
	   already matches what the reference binary does for erasing
	   specifically. This is a real fidelity reduction for strongly biased-
	   composition sequences relative to the reference binary, worth knowing
	   about if your results disagree with running real STREME.

	3. Control generation (only relevant when `control_sequences` is not
	   given): the reference binary's `fasta-shuffle-letters` produces an
	   *exact* k-let-preserving shuffle via a random Eulerian-circuit
	   algorithm. This implementation instead estimates `order`-th order
	   Markov transition probabilities from the primary sequences and
	   *samples* new sequences of matching length from that chain (see
	   `memelite.utils._markov_shuffle_sequences`) -- a standard, much
	   simpler approximation, at the cost of not exactly preserving each
	   individual sequence's own k-mer counts.

	Not implemented: `--bfile` (an externally-supplied background model --
	the background is always estimated from the primary sequences), `--time`
	-based runtime stopping, position-specific priors (`--psp`), and `ttl`
	CLI wiring.


	Parameters
	----------
	sequences: str or list
		The primary sequences to discover motifs in. A FASTA filepath, or a
		list of one-hot numpy arrays (shape (len(alphabet), length), lengths
		may differ) or plain strings.

	control_sequences: str or list or None, optional
		An optional set of control/background sequences, in the same format
		as `sequences`. If None, control sequences are generated by sampling
		from an `order`-th order Markov model estimated from `sequences` (see
		deviation 3 above), one control sequence per primary sequence, each
		matching that primary sequence's length. Default is None.

	alphabet: list or tuple, optional
		A list of characters to use for the alphabet, defining the order that
		characters should appear. Default is ('A', 'C', 'G', 'T').

	minw: int, optional
		The smallest motif width to search. Default is 8.

	maxw: int, optional
		The largest motif width to search. Default is 15.

	order: int, optional
		The Markov order used to estimate the background model for
		automatic control generation (ignored when `control_sequences` is
		given). Default is 2.

	nmotifs: int or None, optional
		If given, search for exactly this many motifs, reporting each
		regardless of significance, rather than stopping based on
		`patience`. Default is None.

	threshold: float, optional
		The p-value (or E-value, if `use_evalue` is True) a motif's held-out
		significance must reach to count as significant. Default is 0.05.

	use_evalue: bool, optional
		Whether `threshold` is compared against the E-value rather than the
		p-value. Default is False.

	patience: int, optional
		The number of consecutive non-significant rounds allowed before
		stopping. Ignored when `nmotifs` is given. Default is 3.

	hofract: float, optional
		The fraction of primary and (independently) control sequences held
		out for the final significance test of each motif, never seen during
		seed search or refinement. Default is 0.1.

	nref: int, optional
		The number of top-scoring candidate seed words, per width, forwarded
		to refinement each round. Default is 4.

	niter: int, optional
		The maximum number of refinement iterations per candidate. Default
		is 20.

	reverse_complement: bool, optional
		Whether to also score the reverse complement strand at each
		position, for both seed scoring and refinement. Default is True.

	eps: float, optional
		A small pseudocount used throughout (seeding PWMs from words,
		re-estimating PWMs from sites, and converting PWMs to log-odds).
		Default is 0.0001.

	random_state: int or None, optional
		A seed governing the primary/control hold-out split and (if
		`control_sequences` is not given) the automatic Markov-shuffle
		control generation. Default is None.

	return_site_positions: bool, optional
		Whether to also return each reported motif's raw per-sequence best-
		site positions/strands (search and held-out primary sequences).
		Default is False.

	n_jobs: int, optional
		The number of threads for numba to use when parallelizing the
		per-sequence scanning kernels. If -1, use all available threads.
		Default is -1.


	Returns
	-------
	motifs: dict[str, numpy.ndarray]
		The discovered PWMs (probabilities, shape (len(alphabet), width)),
		keyed "STREME-1", "STREME-2", etc. in the order found -- the same
		dict shape `read_meme` produces, so this plugs directly into `fimo`,
		`centrimo`, `spamo`, and `write_meme`.

	results: pandas.DataFrame
		One row per motif in `motifs`, with columns `motif_name, motif_idx,
		width, consensus, n_primary_sites, n_control_sites, score_threshold,
		p_value, e_value` (the site counts and p-value are from the held-out
		significance test, not the search phase).

	site_positions: list of dict, optional
		Only returned when `return_site_positions` is True. One dict per
		reported motif, with keys `motif_name, search_primary_positions,
		search_primary_strands, holdout_primary_positions,
		holdout_primary_strands`.
	"""

	if n_jobs != -1:
		_n_jobs = numba.get_num_threads()
		numba.set_num_threads(n_jobs)
	else:
		n_jobs = _n_jobs = numba.get_num_threads()

	if minw > maxw:
		raise ValueError(f"`minw` ({minw}) cannot be greater than `maxw` ({maxw}).")

	alphabet_size = len(alphabet)
	rng = numpy.random.default_rng(random_state)

	_, X_pos_full, Xl_pos_full = _load_ragged_sequences(sequences, alphabet)
	n_pos_total = len(Xl_pos_full) - 1

	if n_pos_total == 0:
		raise ValueError("Cannot run streme with zero sequences.")

	pos_lens = numpy.diff(Xl_pos_full)
	if pos_lens.max() < minw:
		raise ValueError(f"No sequence is at least `minw` ({minw}) bp long "
			f"(the longest is {int(pos_lens.max())} bp).")

	if control_sequences is not None:
		_, X_neg_full, Xl_neg_full = _load_ragged_sequences(control_sequences,
			alphabet)
		n_neg_total = len(Xl_neg_full) - 1

		if n_neg_total == 0:
			raise ValueError("Cannot run streme with zero control sequences.")
	else:
		# Generate controls by shuffling the primary sequences with an
		# `order`-th order Markov model. See deviation 3 in the docstring
		# above for how this differs from the reference binary's exact
		# k-let-preserving shuffle.
		marginal, transitions = _markov_transition_probs(X_pos_full,
			Xl_pos_full, order, alphabet_size, eps)
		seed = int(rng.integers(0, 2**31 - 1))
		X_neg_full = _markov_shuffle_sequences(Xl_pos_full, order,
			alphabet_size, marginal, transitions, seed)
		Xl_neg_full = Xl_pos_full.copy()
		n_neg_total = n_pos_total

	def _split(n_total):
		n_holdout = max(1, int(round(n_total * hofract)))
		if n_total - n_holdout < 1:
			raise ValueError(f"Not enough sequences ({n_total}) to hold out "
				f"a `hofract` ({hofract}) fraction and still have sequences "
				f"left to search.")

		perm = rng.permutation(n_total)
		return perm[n_holdout:], perm[:n_holdout]

	pos_search_idx, pos_holdout_idx = _split(n_pos_total)
	neg_search_idx, neg_holdout_idx = _split(n_neg_total)

	X_pos_search, Xl_pos_search = _subset_ragged(X_pos_full, Xl_pos_full, pos_search_idx)
	X_pos_holdout, Xl_pos_holdout = _subset_ragged(X_pos_full, Xl_pos_full, pos_holdout_idx)
	X_neg_search, Xl_neg_search = _subset_ragged(X_neg_full, Xl_neg_full, neg_search_idx)
	X_neg_holdout, Xl_neg_holdout = _subset_ragged(X_neg_full, Xl_neg_full, neg_holdout_idx)

	n_pos_search, n_neg_search = len(pos_search_idx), len(neg_search_idx)
	n_pos_holdout, n_neg_holdout = len(pos_holdout_idx), len(neg_holdout_idx)

	avg_pos_len = float(pos_lens.mean())
	avg_neg_len = float(numpy.diff(Xl_neg_full).mean())
	avg_w = (minw + maxw) / 2.0

	# The real STREME binary switches from Fisher's exact test to the
	# Binomial test when the primary/control average sequence lengths differ
	# by at least 0.01% (confirmed from source, `streme-utils.c`).
	use_fisher = abs(avg_pos_len - avg_neg_len) < 1e-4 * max(avg_pos_len, avg_neg_len, 1.0)
	search_bernoulli = None if use_fisher else _binomial_bernoulli(
		n_pos_search, n_neg_search, avg_pos_len, avg_neg_len, avg_w)

	columns = ['motif_name', 'motif_idx', 'width', 'consensus',
		'n_primary_sites', 'n_control_sites', 'score_threshold', 'p_value',
		'e_value']

	motifs, rows, site_positions = {}, [], []
	bad_streak = 0

	while True:
		if nmotifs is not None:
			if len(motifs) >= nmotifs:
				break
		elif bad_streak >= patience:
			break

		best, best_w = None, None
		for w in range(minw, maxw + 1):
			codes = _streme_seed_candidates(X_pos_search, Xl_pos_search,
				X_neg_search, Xl_neg_search, w, alphabet_size, nref,
				n_pos_search, n_neg_search, use_fisher, search_bernoulli)

			for code in codes:
				result = _streme_refine(X_pos_search, Xl_pos_search,
					X_neg_search, Xl_neg_search, int(code), w, alphabet_size,
					eps, niter, n_pos_search, n_neg_search, use_fisher,
					search_bernoulli, reverse_complement)

				if best is None or result['p_value'] < best['p_value']:
					best, best_w = result, w

		if best is None:
			break

		w = best_w
		log_pwm_fwd, log_pwm_rc = best['log_pwm_fwd'], best['log_pwm_rc']
		em_threshold = best['threshold']

		# Final significance on the held-out sequences, at the threshold and
		# PWM already fixed during search, so the held-out fraction cannot
		# influence which candidate was chosen.
		ho_pos_scores, ho_pos_pos, ho_pos_strand = _streme_best_sites(
			X_pos_holdout, Xl_pos_holdout, log_pwm_fwd, log_pwm_rc, w,
			reverse_complement)
		ho_neg_scores, ho_neg_pos, ho_neg_strand = _streme_best_sites(
			X_neg_holdout, Xl_neg_holdout, log_pwm_fwd, log_pwm_rc, w,
			reverse_complement)

		ho_pos_count = int((ho_pos_scores >= em_threshold).sum())
		ho_neg_count = int((ho_neg_scores >= em_threshold).sum())

		if use_fisher:
			p_value = float(scipy.stats.hypergeom.sf(ho_pos_count - 1,
				n_pos_holdout + n_neg_holdout, ho_pos_count + ho_neg_count,
				n_pos_holdout))
		else:
			ho_bernoulli = _binomial_bernoulli(n_pos_holdout, n_neg_holdout,
				avg_pos_len, avg_neg_len, avg_w)
			p_value = float(scipy.stats.binom.sf(ho_pos_count - 1,
				ho_pos_count + ho_neg_count, ho_bernoulli))

		# E-value = p-value * number of motifs output so far (including this
		# one), with no per-width/per-seed Bonferroni factor -- confirmed
		# exactly from source, `streme.c`, unlike `centrimo`'s convention.
		n_accepted = len(motifs)
		e_value = min(p_value * (n_accepted + 1), 1.0)

		significant = (e_value if use_evalue else p_value) <= threshold
		forced = nmotifs is not None

		if significant or forced:
			name = f"STREME-{n_accepted + 1}"
			motifs[name] = best['pwm']

			rows.append((name, n_accepted, w,
				characters(best['pwm'], alphabet=alphabet, force=True),
				ho_pos_count, ho_neg_count, em_threshold, p_value, e_value))

			if return_site_positions:
				site_positions.append({
					'motif_name': name,
					'search_primary_positions': best['pos_pos'],
					'search_primary_strands': best['pos_strand'],
					'holdout_primary_positions': ho_pos_pos,
					'holdout_primary_strands': ho_pos_strand,
				})

		bad_streak = 0 if significant else bad_streak + 1

		# Erase this round's winning motif everywhere (search and held-out,
		# primary and control) regardless of significance -- otherwise a
		# locally-best-but-insignificant word would simply be re-found every
		# round, and `patience` would never advance.
		_streme_erase(X_pos_search, Xl_pos_search, best['pos_scores'],
			best['pos_pos'], best['pos_strand'], log_pwm_fwd, log_pwm_rc, w,
			em_threshold)
		_streme_erase(X_neg_search, Xl_neg_search, best['neg_scores'],
			best['neg_pos'], best['neg_strand'], log_pwm_fwd, log_pwm_rc, w,
			em_threshold)
		_streme_erase(X_pos_holdout, Xl_pos_holdout, ho_pos_scores, ho_pos_pos,
			ho_pos_strand, log_pwm_fwd, log_pwm_rc, w, em_threshold)
		_streme_erase(X_neg_holdout, Xl_neg_holdout, ho_neg_scores, ho_neg_pos,
			ho_neg_strand, log_pwm_fwd, log_pwm_rc, w, em_threshold)

	if n_jobs != -1:
		numba.set_num_threads(_n_jobs)

	results = pandas.DataFrame(rows, columns=columns)

	if return_site_positions:
		return motifs, results, site_positions

	return motifs, results
