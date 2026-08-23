# centrimo.py
# Author: Adam He <adamyhe@gmail.com>

import math
import numba
import numpy
import pandas
import scipy.stats

from .io import _fasta_to_flat_array
from .io import _load_motifs
from .utils import _pvalue_score_thresholds


@numba.njit(parallel=True, fastmath=True, cache=True)
def _centrimo_best_sites(X, n_seqs, seq_len, pwm, pwm_lengths, score_thresholds,
	reverse_complement, n_motifs):
	"""An internal function for finding each sequence's best-scoring site.

	This function is the core scanning loop for CentriMo. For each motif and
	each sequence, it finds the single best-scoring site (optionally combining
	the forward and reverse complement strands by taking whichever scores
	higher at a given position) and records that site's signed distance from
	the center of the sequence, along with the raw score it achieved. When
	several positions are tied for the best score, the average of their
	distances is used, mirroring the reference CentriMo implementation.
	Sequences with no site scoring at or above `score_thresholds` have their
	distance marked with NaN and are excluded from that motif's analysis; the
	raw best score is still recorded for every sequence regardless, so that
	callers searching over stricter thresholds (see `centrimo`'s
	`optimize_score`) do not need to re-scan.

	Unlike FIMO's `_fast_hits`, this function does not build a list of hits:
	because CentriMo only ever needs one (tie-averaged) distance per sequence
	per motif, the output is a small, dense `n_motifs` by `n_seqs` array
	rather than a sparse, per-hit structure. This keeps memory use independent
	of how many sequences pass the score threshold, which matters when
	scanning millions of sequences.


	Parameters
	----------
	X: numpy.ndarray, shape=(n_seqs * seq_len,)
		A flat int8 array of alphabet indexes (-1 for unknown characters)
		for all sequences concatenated together. All sequences must have the
		same length.

	n_seqs: int
		The number of sequences encoded in `X`.

	seq_len: int
		The length of each sequence.

	pwm: numpy.ndarray, shape=(len(alphabet), total_width)
		The concatenated log-odds PWMs. When `reverse_complement` is True,
		this contains the forward PWMs followed by their reverse complements
		in the same order.

	pwm_lengths: numpy.ndarray
		The cumulative offsets demarcating each PWM's span within `pwm`. Has
		`2 * n_motifs + 1` entries when `reverse_complement` is True, and
		`n_motifs + 1` entries otherwise.

	score_thresholds: numpy.ndarray, shape=(n_motifs,)
		The raw score threshold for each motif. This threshold is shared
		across strands because the reverse complement of a PWM has an
		identical score distribution under a uniform background.

	reverse_complement: bool
		Whether to also score the reverse complement strand at each position
		and combine it with the forward strand by taking the higher score.

	n_motifs: int
		The number of (merged) motifs being scanned.


	Returns
	-------
	distances: numpy.ndarray, shape=(n_motifs, n_seqs)
		The signed distance from the sequence center of each sequence's
		best-scoring site, or NaN if no site reached `score_thresholds`.

	best_scores: numpy.ndarray, shape=(n_motifs, n_seqs)
		The raw score of each sequence's best-scoring site, recorded
		unconditionally (i.e. even for sequences below `score_thresholds`).
	"""

	distances = numpy.empty((n_motifs, n_seqs), dtype=numpy.float64)
	distances[:] = numpy.nan
	best_scores = numpy.empty((n_motifs, n_seqs), dtype=numpy.float64)

	half = (seq_len - 1) / 2.0

	for k in numba.prange(n_motifs):
		w = pwm_lengths[k+1] - pwm_lengths[k]
		thresh = score_thresholds[k]
		s0 = pwm_lengths[k]
		n_valid = seq_len - w + 1

		if reverse_complement:
			rc0 = pwm_lengths[k + n_motifs]

		for s in range(n_seqs):
			base = s * seq_len
			best_score = -1e300
			sum_centers = 0.0
			tie_count = 0

			for i in range(n_valid):
				score = 0.0
				for j in range(w):
					idx = X[base + i + j]
					if idx == -1:
						continue
					score += pwm[idx, s0 + j]

				if reverse_complement:
					score_rc = 0.0
					for j in range(w):
						idx = X[base + i + j]
						if idx == -1:
							continue
						score_rc += pwm[idx, rc0 + j]

					if score_rc > score:
						score = score_rc

				center = i + (w - 1) / 2.0
				if score > best_score:
					best_score = score
					sum_centers = center
					tie_count = 1
				elif score == best_score:
					sum_centers += center
					tie_count += 1

			best_scores[k, s] = best_score
			if best_score >= thresh:
				distances[k, s] = sum_centers / tie_count - half

	return distances, best_scores


def _load_sequences(sequences, alphabet, seqlen):
	"""An internal function for loading a set of equal-length sequences.

	This function accepts either a FASTA filepath or a one-hot numpy array
	and returns a flat int8 representation (shared by `_centrimo_best_sites`
	and `spamo`'s kernels), along with the number and length of the
	sequences it contains. When `sequences` is a FASTA filepath, only
	sequences matching `seqlen` (or the length of the first sequence in the
	file, if `seqlen` is None) are kept, mirroring the reference CentriMo
	binary's own default `--seqlen` behavior rather than requiring the
	caller to pre-filter the file.
	"""

	if isinstance(sequences, str):
		_, X_flat, X_lengths = _fasta_to_flat_array(sequences, alphabet)
		seq_lens = numpy.diff(X_lengths)

		target_len = seqlen if seqlen is not None else int(seq_lens[0])
		keep = numpy.where(seq_lens == target_len)[0]
		n_dropped = len(seq_lens) - len(keep)

		if n_dropped > 0:
			print(f"ignoring {n_dropped} sequence(s) not of length "
				f"{target_len} (use `seqlen` to select a different length).")

		seq_len = target_len
		n_seqs = len(keep)
		X = numpy.empty(n_seqs * seq_len, dtype=numpy.int8)
		for out_i, in_i in enumerate(keep):
			X[out_i*seq_len:(out_i+1)*seq_len] = X_flat[X_lengths[in_i]:X_lengths[in_i+1]]

	else:
		if not isinstance(sequences, numpy.ndarray):
			sequences = sequences.numpy()

		n_seqs, _, seq_len = sequences.shape
		X = ((sequences.argmax(axis=1) + 1) * sequences.sum(axis=1)) - 1
		X = X.astype(numpy.int8).flatten()

	return X, n_seqs, seq_len


def centrimo(motifs, sequences, control_sequences=None,
	alphabet=['A', 'C', 'G', 'T'], bin_size=0.1,
	eps=0.0001, threshold=0.001, min_width=1, max_width=None, width_step=2,
	window_widths=None, reverse_complement=True, separate_strands=False,
	flip=False, optimize_score=False, max_score_thresholds=50, seqlen=None,
	return_site_distances=False, n_jobs=-1):
	"""An implementation of the CentriMo algorithm from the MEME suite.

	This function implements the "Central Motif Enrichment Analysis"
	(CentriMo) algorithm from the MEME suite. Given a set of equal-length
	sequences (e.g., ChIP-seq peaks or other windows centered on a feature of
	interest), CentriMo tests, for each motif, whether that motif's best site
	in each sequence is enriched near the sequence center rather than
	uniformly distributed across it. This is useful for confirming that a
	motif discovered or scanned some other way (e.g., via `fimo`, or pulled
	out of a machine learning model's attributions) is likely the true
	binding determinant rather than an incidental look-alike.

	For each motif, each sequence is reduced to a single number: the signed
	distance from the sequence's center of that sequence's best-scoring site
	(ties are averaged; both strands are considered by default and combined
	by taking whichever scores higher at a given position). Sequences with no
	site scoring at or above the threshold are excluded from that motif's
	analysis entirely. A one-sided binomial test is then run for a range of
	window widths centered on the sequence midpoint, comparing the number of
	sequences whose best site falls in the window to what would be expected
	if best sites were uniformly distributed; the window with the smallest
	p-value is reported, Bonferroni-corrected for the number of window widths
	and the number of motifs tested.

	If `control_sequences` is given, this also runs the reference CentriMo
	binary's `--neg` differential/comparative mode: the enriched window for
	each motif is still selected using only the primary `sequences` (exactly
	as above, so the control set cannot bias which window is chosen), and
	then a one-sided Fisher's exact test compares, at that fixed window, the
	number of primary vs. control sequences whose best site falls inside it
	versus outside it. This answers a different question than the one-sample
	test: not "is this motif centrally enriched at all," but "is it more
	centrally enriched in the primary set than in the control set" (e.g.
	bound vs. unbound peaks, or real vs. shuffled sequences). The primary and
	control sets do not need to have the same length or number of sequences,
	since each is reduced to its own center-relative distances independently
	before the fixed window (a bp radius) is applied to both.

	If `optimize_score` is True, this also runs the reference CentriMo
	binary's `--optimize_score` mode: rather than fixing the score threshold
	at the value implied by `threshold`, a range of stricter thresholds is
	also searched (every distinct score actually achieved by some sequence's
	best site, at or above the `threshold`-implied minimum, capped at
	`max_score_thresholds` distinct values) and combined jointly with the
	window-width search, so both the window and the threshold reported are
	whichever combination gives the smallest p-value. This can find sharper
	enrichment when only the sequences with the very strongest matches are
	truly centrally enriched, at the cost of an additional Bonferroni factor
	for the number of thresholds tried. Since the qualifying sequence set can
	now shrink beyond what `threshold` alone implies, `n_sequences` and
	`n_matching_sequences` reflect the optimized threshold rather than the
	nominal one, and an `optimized_threshold_p_value` column reports the
	p-value equivalent of the threshold that was actually selected. When
	combined with `control_sequences`, the threshold (like the window) is
	still selected using only the primary sequences before the control set is
	consulted at all.

	`flip` mirrors the reference CentriMo binary's `--flip` option: reverse
	complement matches are reported "reflected" around the sequence center
	(i.e. with their sign negated) rather than at their literal position in
	the given sequence's forward-strand coordinates. This is a purely
	presentational convention with no effect on any statistic in the returned
	dataframe (every test here is computed from the *absolute* distance from
	center, which a sign flip does not change) -- it only affects the sign of
	the `-rc` rows within the raw `distances`/`control_distances` arrays
	returned when `return_site_distances` is True, e.g. for building your own
	site-probability plot. Because there is no well-defined per-sequence
	strand identity to reflect once forward and reverse complement scores
	have been combined into one row, `flip` requires `separate_strands=True`.

	Note that this implementation always requires equal-length sequences
	(within each of `sequences` and `control_sequences` separately) and
	exposes the match threshold as a p-value (converted internally to a raw
	score threshold, as in `fimo`) rather than the reference CentriMo binary's
	fixed-bits `--score` option, for consistency with the rest of this
	package.


	Parameters
	----------
	motifs: str or dict
		A MEME file to load containing motifs to scan, or a dictionary where
		the keys are names of motifs and the values are PWMs with shape
		(len(alphabet), pwm_length).

	sequences: str or numpy.ndarray
		A set of equal-length sequences to scan the motifs against. If this
		is a string, assumes it is a filepath to a FASTA-formatted file, and
		only sequences matching the length of the first sequence (or
		`seqlen`, if provided) will be used. If this is a numpy array, it
		must have shape (n_sequences, len(alphabet), sequence_length).

	control_sequences: str or numpy.ndarray or None, optional
		An optional set of control/background sequences (e.g. unbound peaks
		or shuffled sequences), in the same format as `sequences`, that
		triggers the differential (`--neg`-style) comparison described
		above. Its sequences must be equal-length among themselves, but need
		not match the length or count of `sequences`. Default is None.

	alphabet: list, optional
		A list of characters to use for the alphabet, defining the order that
		characters should appear. Default is ['A', 'C', 'G', 'T'].

	bin_size: float, optional
		The size of the bins discretizing the PWM scores when converting the
		p-value threshold to a raw score threshold. Default is 0.1.

	eps: float, optional
		A small pseudocount to add to the motif PWMs before taking the log.
		Default is 0.0001.

	threshold: float, optional
		The p-value threshold a site must reach to be considered a sequence's
		best site. Sequences with no site reaching this threshold are
		excluded from that motif's analysis. Default is 0.001.

	min_width: int, optional
		The smallest window width to test. Default is 1.

	max_width: int or None, optional
		The largest window width to test. If None, uses the sequence length.
		Default is None.

	width_step: int, optional
		The step size between tested window widths. The default of 2, paired
		with the default `min_width` of 1, tests only odd widths, which keeps
		each window exactly symmetric around the sequence center. Default is
		2.

	window_widths: list or numpy.ndarray or None, optional
		An explicit set of window widths to test, overriding `min_width`,
		`max_width`, and `width_step`. Default is None.

	reverse_complement: bool, optional
		Whether to also score the reverse complement strand at each position,
		combining it with the forward strand by taking whichever scores
		higher (unless `separate_strands` is True). Default is True.

	separate_strands: bool, optional
		By default, when `reverse_complement` is True, the forward and
		reverse complement strands are combined into a single reported
		motif (whichever strand scores higher at a given position wins).
		Setting this to True instead reports each strand as its own row,
		named `{motif_name}` and `{motif_name}-rc`, each independently
		scanned, windowed, and tested -- mirroring the reference CentriMo
		binary's `--sep` option. Has no effect when `reverse_complement` is
		False, since there is then no separate strand to report. Default is
		False.

	flip: bool, optional
		Whether to report `-rc` rows' site distances reflected around the
		sequence center (sign-negated) rather than at their literal
		forward-strand position, mirroring the reference CentriMo binary's
		`--flip` option. Purely presentational: see the extended description
		above. Requires `separate_strands=True`. Default is False.

	optimize_score: bool, optional
		Whether to also search over stricter score thresholds (in addition
		to the window-width search), reporting whichever (threshold, window)
		combination is most significant, mirroring the reference CentriMo
		binary's `--optimize_score` option. See the extended description
		above. Default is False.

	max_score_thresholds: int, optional
		The maximum number of distinct score thresholds to test when
		`optimize_score` is True. If more distinct scores are actually
		achieved than this, an evenly-spaced subset is used instead, to
		bound both the search cost and the Bonferroni penalty it incurs.
		Ignored when `optimize_score` is False. Default is 50.

	seqlen: int or None, optional
		When `sequences` is a FASTA filepath, only sequences of this length
		are used; sequences of any other length are ignored. If None, uses
		the length of the first sequence in the file. Ignored when
		`sequences` is a numpy array. Default is None.

	return_site_distances: bool, optional
		Whether to also return the raw per-sequence, per-motif distances
		used to compute the enrichment statistics. Default is False.

	n_jobs: int, optional
		The number of threads for numba to use when parallelizing the
		processing of motifs. If -1, use all available threads. Default is
		-1.


	Returns
	-------
	results: pandas.DataFrame
		A dataframe with one row per motif, containing the number of
		sequences used, the best window found, and the corresponding
		enrichment statistics. When `control_sequences` is given, this also
		includes the control set's sequence/match counts at that window and
		the Fisher's exact test p-value/E-value for the differential
		comparison. When `optimize_score` is given, this also includes the
		p-value equivalent of the score threshold that was selected.

	distances: numpy.ndarray, shape=(n_motifs, n_sequences), optional
		Only returned when `return_site_distances` is True. The signed
		distance from the sequence center of each sequence's best site for
		each motif, or NaN if no site reached the threshold.

	control_distances: numpy.ndarray, shape=(n_motifs, n_control_sequences), optional
		Only returned when `return_site_distances` is True and
		`control_sequences` was given. The same as `distances`, but for the
		control sequences.
	"""

	if n_jobs != -1:
		_n_jobs = numba.get_num_threads()
		numba.set_num_threads(n_jobs)
	else:
		n_jobs = _n_jobs = numba.get_num_threads()

	if flip and not separate_strands:
		raise ValueError("`flip` requires `separate_strands=True`, since "
			"there is no well-defined per-sequence strand identity to "
			"reflect once forward and reverse complement scores have been "
			"combined into a single row.")

	has_control = control_sequences is not None

	columns = ['motif_name', 'motif_idx', 'width', 'n_sequences',
		'n_valid_positions', 'best_window_width', 'n_matching_sequences',
		'p_value', 'e_value']

	if optimize_score:
		columns += ['optimized_threshold_p_value']

	if has_control:
		columns += ['n_control_sequences', 'n_control_matching_sequences',
			'fisher_p_value', 'fisher_e_value']

	# Extract the motifs
	names, pwms = _load_motifs(motifs)
	n_motifs = len(names)

	if n_motifs == 0:
		if n_jobs != -1:
			numba.set_num_threads(_n_jobs)
		return pandas.DataFrame(columns=columns)

	if separate_strands and reverse_complement:
		# Expand each motif into its own forward and reverse complement
		# entries up front, then scan the rest of the pipeline with
		# `reverse_complement=False`: each entry is now single-stranded, so
		# there is nothing left to combine at the kernel level, and every
		# downstream step (window search, E-value, the differential Fisher
		# test) runs identically per entry, exactly as it would for any two
		# independent motifs.
		names = names + [name + "-rc" for name in names]
		pwms = pwms + [pwm[::-1, ::-1] for pwm in pwms]
		n_motifs = len(names)
		reverse_complement = False

	widths = numpy.array([pwm.shape[-1] for pwm in pwms], dtype=numpy.int64)
	fwd_lengths = numpy.cumsum([0] + list(widths)).astype(numpy.int64)
	fwd_concat = numpy.concatenate(pwms, axis=-1)

	if reverse_complement:
		rc_pwms = [pwm[::-1, ::-1] for pwm in pwms]
		all_concat = numpy.concatenate([fwd_concat] + rc_pwms, axis=-1)
		rc_lengths = fwd_lengths[1:] + fwd_lengths[-1]
		pwm_lengths = numpy.concatenate([fwd_lengths, rc_lengths]).astype(numpy.int64)
	else:
		all_concat = fwd_concat
		pwm_lengths = fwd_lengths

	log_pwm = numpy.log2(all_concat + eps) - math.log2(0.25)

	# Convert the p-value threshold to a per-motif raw score threshold. The
	# same threshold is used for both strands since the reverse complement of
	# a PWM has an identical score distribution under a uniform background.
	score_thresholds, _smallest, _score_to_pvals = _pvalue_score_thresholds(
		log_pwm[:, :fwd_lengths[-1]], fwd_lengths, bin_size, threshold)

	# Extract the sequences, requiring that they all have the same length
	X, n_seqs, seq_len = _load_sequences(sequences, alphabet, seqlen)

	if n_seqs == 0:
		raise ValueError("Cannot run centrimo with zero sequences.")

	if widths.max() > seq_len:
		bad = names[int(widths.argmax())]
		raise ValueError(f"Motif '{bad}' (width {int(widths.max())}) is wider "
			f"than the sequence length ({seq_len}).")

	if has_control:
		X_neg, n_neg_seqs, neg_seq_len = _load_sequences(control_sequences,
			alphabet, seqlen)

		if n_neg_seqs == 0:
			raise ValueError("Cannot run centrimo with zero control sequences.")

		if widths.max() > neg_seq_len:
			bad = names[int(widths.argmax())]
			raise ValueError(f"Motif '{bad}' (width {int(widths.max())}) is "
				f"wider than the control sequence length ({neg_seq_len}).")

	distances, best_scores = _centrimo_best_sites(X, n_seqs, seq_len, log_pwm,
		pwm_lengths, score_thresholds, reverse_complement, n_motifs)

	if has_control:
		control_distances, control_best_scores = _centrimo_best_sites(X_neg,
			n_neg_seqs, neg_seq_len, log_pwm, pwm_lengths, score_thresholds,
			reverse_complement, n_motifs)

	if n_jobs != -1:
		numba.set_num_threads(_n_jobs)

	if flip:
		# Purely presentational: reflects each `-rc` row's site distances
		# around the sequence center. Every statistic below is computed from
		# `abs(distance)`, which a sign flip leaves unchanged, so this only
		# affects the raw arrays returned by `return_site_distances`.
		rc_rows = numpy.array([name.endswith("-rc") for name in names])
		distances[rc_rows] *= -1
		if has_control:
			control_distances[rc_rows] *= -1

	# Compute the enrichment statistics for each motif, testing a range of
	# window widths (and, if `optimize_score`, score thresholds) and
	# correcting for the number tested. This operates on the small
	# `distances`/`best_scores` arrays only, and so is not a throughput
	# bottleneck.
	rows = []
	for k in range(n_motifs):
		w = int(widths[k])
		n_valid_positions = seq_len - w + 1

		valid = ~numpy.isnan(distances[k])
		n_nominal = int(valid.sum())

		if window_widths is not None:
			cand_widths = numpy.asarray(window_widths)
		else:
			hi = max_width if max_width is not None else seq_len
			cand_widths = numpy.arange(min_width, hi + 1, width_step)

		cand_widths = cand_widths[(cand_widths >= 1) &
			(cand_widths <= n_valid_positions)]
		radii = ((cand_widths - 1) // 2).astype(numpy.float64)
		p_null = cand_widths / n_valid_positions

		if has_control:
			neg_valid = ~numpy.isnan(control_distances[k])
			n_neg_nominal = int(neg_valid.sum())

		if n_nominal == 0 or len(cand_widths) == 0:
			row = (names[k], k, w, n_nominal, n_valid_positions, numpy.nan,
				0, 1.0, 1.0)
			if optimize_score:
				row += (1.0,)
			if has_control:
				row += (n_neg_nominal, 0, 1.0, 1.0)
			rows.append(row)
			continue

		abs_d = numpy.abs(distances[k, valid])

		if not optimize_score:
			sorted_abs_d = numpy.sort(abs_d)
			counts = numpy.searchsorted(sorted_abs_d, radii, side='right')
			p_values = scipy.stats.binom.sf(counts - 1, n_nominal, p_null)

			best = int(numpy.argmin(p_values))
			n = n_nominal
			best_width = int(cand_widths[best])
			k_pos = int(counts[best])
			r_star = radii[best]
			t_star = None
			p_value = float(p_values[best])
			e_value = min(p_value * len(cand_widths) * n_motifs, 1.0)
		else:
			# Search jointly over score thresholds (every distinct score
			# actually achieved by some sequence's best site, at or above
			# the `threshold`-implied minimum already baked into `valid`)
			# and window widths, reporting whichever combination is most
			# significant.
			scores_valid = best_scores[k, valid]
			grid = numpy.unique(scores_valid)

			if len(grid) > max_score_thresholds:
				idx = numpy.unique(numpy.round(numpy.linspace(
					0, len(grid) - 1, max_score_thresholds)).astype(numpy.int64))
				grid = grid[idx]

			best_p, best_width, k_pos, n, t_star, r_star = (numpy.inf,
				int(cand_widths[0]), 0, 0, grid[0], radii[0])

			for t in grid:
				sub = scores_valid >= t
				n_t = int(sub.sum())
				if n_t == 0:
					continue

				sorted_abs_d_t = numpy.sort(abs_d[sub])
				counts_t = numpy.searchsorted(sorted_abs_d_t, radii, side='right')
				p_values_t = scipy.stats.binom.sf(counts_t - 1, n_t, p_null)

				local_best = int(numpy.argmin(p_values_t))
				if p_values_t[local_best] < best_p:
					best_p = p_values_t[local_best]
					best_width = int(cand_widths[local_best])
					k_pos = int(counts_t[local_best])
					n = n_t
					t_star = t
					r_star = radii[local_best]

			p_value = float(best_p)
			e_value = min(p_value * len(grid) * len(cand_widths) * n_motifs, 1.0)

			bin_idx = int(round(t_star / bin_size)) - int(_smallest[k])
			bin_idx = min(max(bin_idx, 0), len(_score_to_pvals[k]) - 1)
			optimized_threshold_p_value = float(2.0 ** _score_to_pvals[k][bin_idx])

		row = (names[k], k, w, n, n_valid_positions, best_width, k_pos,
			p_value, float(e_value))

		if optimize_score:
			row += (optimized_threshold_p_value,)

		if has_control:
			# The window (and, if `optimize_score`, the threshold) was
			# selected using only the primary sequences above, so the
			# control set cannot bias which one gets tested here.
			if optimize_score:
				neg_sub = control_best_scores[k, neg_valid] >= t_star
				n_neg = int(neg_sub.sum())
				abs_d_neg = numpy.abs(control_distances[k, neg_valid])[neg_sub]
			else:
				n_neg = n_neg_nominal
				abs_d_neg = numpy.abs(control_distances[k, neg_valid])

			k_neg = int((abs_d_neg <= r_star).sum())

			if n_neg == 0:
				fisher_p = 1.0
			else:
				table = [[k_pos, n - k_pos], [k_neg, n_neg - k_neg]]
				_, fisher_p = scipy.stats.fisher_exact(table, alternative='greater')

			fisher_e = min(fisher_p * n_motifs, 1.0)
			row += (n_neg, k_neg, float(fisher_p), float(fisher_e))

		rows.append(row)

	result = pandas.DataFrame(rows, columns=columns)

	if return_site_distances:
		if has_control:
			return result, distances, control_distances
		return result, distances

	return result
