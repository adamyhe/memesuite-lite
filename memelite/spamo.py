# spamo.py
# Author: Adam He <adamyhe@gmail.com>

import math
import numba
import numpy
import pandas
import scipy.stats

from .io import _load_motifs
from .utils import _pvalue_score_thresholds
from .centrimo import _load_sequences


@numba.njit(parallel=True, fastmath=True, cache=True)
def _spamo_primary_sites(X, n_seqs, seq_len, pwm_fwd, pwm_rc, w_p,
	score_threshold, margin, reverse_complement):
	"""An internal function for finding each sequence's best primary site.

	For each sequence, finds the single best-scoring primary motif site
	within the *central* region only -- excluding `margin` bp from each
	edge -- so that a full `margin`-wide flanking window always fits inside
	the sequence around wherever the primary site ends up. Both strands are
	considered when `reverse_complement` is True, and the winning strand is
	recorded alongside the position, since `spamo` needs to know whether a
	secondary site shares the primary's strand or not.

	Unlike `centrimo`'s tie-averaging (which averages *distances*, a
	meaningful thing to average), ties here are broken deterministically by
	keeping the first (leftmost) position encountered, preferring the
	forward strand on an exact forward/rc tie at the same position (since
	the forward score is checked first) -- averaging two different absolute
	*positions*, or two different strand identities, would not correspond to
	anything meaningful.


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

	pwm_fwd: numpy.ndarray, shape=(len(alphabet), w_p)
		The primary motif's log-odds PWM.

	pwm_rc: numpy.ndarray, shape=(len(alphabet), w_p)
		The reverse complement of `pwm_fwd`. Read even when
		`reverse_complement` is False, so the caller must still supply it.

	w_p: int
		The width of the primary motif.

	score_threshold: float
		The raw score a site must reach to qualify.

	margin: int
		The number of bp to exclude from each edge of the sequence when
		scanning for the primary site.

	reverse_complement: bool
		Whether to also score the reverse complement strand at each
		position and keep whichever strand scores higher.


	Returns
	-------
	positions: numpy.ndarray, shape=(n_seqs,), dtype=int32
		The start position of each sequence's best primary site, or -1 if no
		site reached `score_threshold`.

	strands: numpy.ndarray, shape=(n_seqs,), dtype=int8
		0 if the winning site was on the forward strand, 1 if reverse
		complement. Meaningless where `positions == -1`.
	"""

	positions = numpy.full(n_seqs, -1, dtype=numpy.int32)
	strands = numpy.zeros(n_seqs, dtype=numpy.int8)

	lo = margin
	hi = seq_len - margin - w_p

	for s in numba.prange(n_seqs):
		base = s * seq_len
		best_score = -1e300
		best_i = -1
		best_strand = 0

		for i in range(lo, hi + 1):
			score = 0.0
			for j in range(w_p):
				idx = X[base + i + j]
				if idx == -1:
					continue
				score += pwm_fwd[idx, j]

			if score > best_score:
				best_score = score
				best_i = i
				best_strand = 0

			if reverse_complement:
				score_rc = 0.0
				for j in range(w_p):
					idx = X[base + i + j]
					if idx == -1:
						continue
					score_rc += pwm_rc[idx, j]

				if score_rc > best_score:
					best_score = score_rc
					best_i = i
					best_strand = 1

		if best_score >= score_threshold:
			positions[s] = best_i
			strands[s] = best_strand

	return positions, strands


@numba.njit(parallel=True, fastmath=True, cache=True)
def _spamo_secondary_sites(X, n_seqs, seq_len, primary_positions,
	primary_strands, w_p, pwm, pwm_lengths, score_thresholds, margin,
	reverse_complement, n_motifs):
	"""An internal function for finding each sequence's best secondary site.

	For each secondary motif and each sequence with a qualifying primary
	site at position `p` (from `_spamo_primary_sites`), finds the single
	best-scoring secondary site within `margin` bp on either side of `p`,
	skipping any candidate site that overlaps the primary's own span
	`[p, p+w_p)` (so the two sites can never overlap). Sequences with no
	qualifying primary site are skipped entirely. Otherwise identical
	scanning/thresholding/tie-breaking logic to `_spamo_primary_sites`.


	Parameters
	----------
	X, n_seqs, seq_len: see `_spamo_primary_sites`.

	primary_positions: numpy.ndarray, shape=(n_seqs,)
		The primary site position for each sequence, from
		`_spamo_primary_sites` (-1 = no qualifying primary site).

	primary_strands: numpy.ndarray, shape=(n_seqs,)
		The primary site's winning strand for each sequence, from
		`_spamo_primary_sites`.

	w_p: int
		The width of the primary motif.

	pwm: numpy.ndarray, shape=(len(alphabet), total_width)
		The concatenated secondary motif log-odds PWMs. When
		`reverse_complement` is True, this contains the forward PWMs
		followed by their reverse complements in the same order (the same
		convention as `centrimo._centrimo_best_sites`).

	pwm_lengths: numpy.ndarray
		The cumulative offsets demarcating each secondary PWM's span within
		`pwm`. Has `2 * n_motifs + 1` entries when `reverse_complement` is
		True, and `n_motifs + 1` entries otherwise.

	score_thresholds: numpy.ndarray, shape=(n_motifs,)
		The raw score threshold for each secondary motif.

	margin: int
		The number of bp on either side of the primary site to search.

	reverse_complement: bool
		Whether to also score the reverse complement strand.

	n_motifs: int
		The number of secondary motifs being scanned.


	Returns
	-------
	positions: numpy.ndarray, shape=(n_motifs, n_seqs), dtype=int32
		The start position of each sequence's best secondary site for each
		motif, or -1 if no site reached `score_thresholds` (or the sequence
		had no qualifying primary site at all).

	strands: numpy.ndarray, shape=(n_motifs, n_seqs), dtype=int8
		0 if the winning secondary site was on the forward strand, 1 if
		reverse complement. Meaningless where `positions == -1`.
	"""

	positions = numpy.full((n_motifs, n_seqs), -1, dtype=numpy.int32)
	strands = numpy.zeros((n_motifs, n_seqs), dtype=numpy.int8)

	for k in numba.prange(n_motifs):
		w_s = pwm_lengths[k+1] - pwm_lengths[k]
		s0 = pwm_lengths[k]
		thresh = score_thresholds[k]

		if reverse_complement:
			rc0 = pwm_lengths[k + n_motifs]

		for s in range(n_seqs):
			p = primary_positions[s]
			if p == -1:
				continue

			base = s * seq_len
			lo = max(p - margin, 0)
			hi = min(p + w_p + margin - w_s, seq_len - w_s)

			best_score = -1e300
			best_i = -1
			best_strand = 0

			for i in range(lo, hi + 1):
				if i + w_s > p and i < p + w_p:
					continue

				score = 0.0
				for j in range(w_s):
					idx = X[base + i + j]
					if idx == -1:
						continue
					score += pwm[idx, s0 + j]

				if score > best_score:
					best_score = score
					best_i = i
					best_strand = 0

				if reverse_complement:
					score_rc = 0.0
					for j in range(w_s):
						idx = X[base + i + j]
						if idx == -1:
							continue
						score_rc += pwm[idx, rc0 + j]

					if score_rc > best_score:
						best_score = score_rc
						best_i = i
						best_strand = 1

			if best_score >= thresh:
				positions[k, s] = best_i
				strands[k, s] = best_strand

	return positions, strands


def spamo(primary_motif, secondary_motifs, sequences,
	alphabet=('A', 'C', 'G', 'T'), margin=150, range_=None, bin_size_bp=1,
	bin_size=0.1, eps=0.0001, threshold=0.001, reverse_complement=True,
	seqlen=None, return_site_positions=False, n_jobs=-1):
	"""An implementation of the SpaMo algorithm from the MEME suite.

	This function implements the "Spaced Motif Analysis" (SpaMo) algorithm
	from the MEME suite. Given a set of equal-length sequences, a single
	"primary" motif, and a database of "secondary" motifs, SpaMo tests
	whether each secondary motif's best site tends to occur at a specific
	distance and orientation relative to the primary motif's best site --
	evidence that the two motifs' binding factors physically interact. This
	is useful for confirming or discovering motif-pair spacing/orientation
	constraints (motif "syntax"), e.g. between motifs pulled out of a
	machine learning model's attributions.

	For each sequence, the primary motif's single best site is found within
	the *central* region only (excluding `margin` bp from each edge, so a
	full flanking window always fits around wherever it lands). For each
	sequence with a qualifying primary site, each secondary motif's single
	best site is then found within `margin` bp on either side of the
	primary site, excluding the primary's own span (the two sites can never
	overlap). The signed offset `f` between the two sites is computed as
	`f = -(D+1)` if the secondary is 5' (upstream) of the primary, else
	`f = D+1`, where `D` is the gap in bp between the closest edges of the
	two sites (so `f` is never 0). Sequences are split into two entirely
	independent groups per secondary motif -- "same strand" and "opposite
	strand", depending on whether the primary and secondary sites landed on
	the same strand -- each with its own histogram and test, since this is
	itself an interesting distinction (some interactions are orientation-
	specific) and not just a nuisance to average over.

	For each `(secondary motif, strand category)`, a one-sided binomial test
	is run for every bin of width `bin_size_bp` within `range_` bp on either
	side of the primary (`N` = sequences with a valid offset in that
	category, `s` = count in a bin, `q = bin_size_bp / (2r+1)` where
	`r = (2*margin - w_p - w_s)/2` is a uniform-null probability -- the same
	idea as `centrimo`'s `p_null`). Because `q` is identical for every bin
	here (fixed width, fixed `N`), the bin with the highest count is always
	the most significant one, so only a single `scipy.stats.binom.sf` call
	is needed per `(secondary motif, strand category)`, unlike `centrimo`'s
	window-width search. The reported E-value Bonferroni-corrects for the
	number of bins tested, the two strand categories, and the number of
	secondary motifs.

	Only the best secondary site per sequence is used (matching the
	reference SpaMo binary's `-usebestsec` option, but as the *only*
	behavior here rather than a non-default flag): the underlying binomial
	test assumes exactly one offset value per sequence, which is only
	statistically valid under best-site semantics -- the reference binary's
	default "count all matches above threshold" mode would let one sequence
	contribute multiple offsets, violating that assumption. This mirrors
	`centrimo`'s identical, previously-made choice in this package.

	Also not implemented: `-trim` (motif-edge information-content trimming
	-- apply it to your PWMs yourself before calling `spamo`, if wanted),
	`-shared`/`-overlap`/`-joint` (secondary-motif redundancy clustering),
	multiple named secondary-motif databases (pass one flat dict), and
	`-dumpseqs`. As in `fimo`/`centrimo`, `threshold` is a p-value (applied
	to both the primary and secondary motifs), converted internally to a raw
	score threshold, rather than the reference binary's fixed-bits
	`-minscore`.


	Parameters
	----------
	primary_motif: str or dict
		A MEME file or dict (matching `fimo`/`centrimo`'s `motifs` contract)
		that must resolve to exactly one motif. If your dict has more than
		one entry, index into it yourself first.

	secondary_motifs: str or dict
		A MEME file to load containing motifs to scan, or a dictionary where
		the keys are names of motifs and the values are PWMs with shape
		(len(alphabet), pwm_length). Any number of motifs.

	sequences: str or numpy.ndarray
		A set of equal-length sequences, in the same format as `centrimo`'s
		`sequences` (a FASTA filepath or a one-hot array of shape
		(n_sequences, len(alphabet), sequence_length)).

	alphabet: list or tuple, optional
		A list of characters to use for the alphabet, defining the order that
		characters should appear. Default is ('A', 'C', 'G', 'T').

	margin: int, optional
		The number of bp to exclude from each edge of the sequence when
		scanning for the primary site, and the radius (in bp) on either side
		of the primary site to search for secondary sites. Default is 150.

	range_: int or None, optional
		How far out (bp, on either side of the primary site) the candidate
		offset bins for the significance search extend. If None, uses
		`margin`. Sequences with an offset beyond `range_` (but still within
		`margin`) still count toward a motif's `n_sequences`, but are not
		considered when searching for the most significant bin. Default is
		None.

	bin_size_bp: int, optional
		The width, in bp, of each candidate offset bin. Default is 1.

	bin_size: float, optional
		The size of the bins discretizing PWM scores when converting the
		p-value threshold to a raw score threshold (as in `fimo`/`centrimo`
		-- unrelated to `bin_size_bp`). Default is 0.1.

	eps: float, optional
		A small pseudocount to add to the motif PWMs before taking the log.
		Default is 0.0001.

	threshold: float, optional
		The p-value threshold a site (primary or secondary) must reach to
		qualify. Default is 0.001.

	reverse_complement: bool, optional
		Whether to also score the reverse complement strand at each
		position, for both the primary and secondary scans. Default is
		True.

	seqlen: int or None, optional
		When `sequences` is a FASTA filepath, only sequences of this length
		are used; sequences of any other length are ignored. If None, uses
		the length of the first sequence in the file. Default is None.

	return_site_positions: bool, optional
		Whether to also return the raw per-sequence primary/secondary site
		positions and strands. Default is False.

	n_jobs: int, optional
		The number of threads for numba to use when parallelizing the
		processing of secondary motifs. If -1, use all available threads.
		Default is -1.


	Returns
	-------
	results: pandas.DataFrame
		A dataframe with two rows per secondary motif (`strand` == 'same'
		then 'opposite'), containing the number of sequences used, the most
		significant offset bin found, and the corresponding enrichment
		statistics.

	primary_positions: numpy.ndarray, shape=(n_sequences,), optional
	primary_strands: numpy.ndarray, shape=(n_sequences,), optional
	secondary_positions: numpy.ndarray, shape=(n_motifs, n_sequences), optional
	secondary_strands: numpy.ndarray, shape=(n_motifs, n_sequences), optional
		Only returned when `return_site_positions` is True.
	"""

	if n_jobs != -1:
		_n_jobs = numba.get_num_threads()
		numba.set_num_threads(n_jobs)
	else:
		n_jobs = _n_jobs = numba.get_num_threads()

	if range_ is None:
		range_ = margin

	columns = ['motif_name', 'motif_idx', 'width', 'strand', 'n_sequences',
		'n_bins_tested', 'best_offset_lo', 'best_offset_hi',
		'n_matching_sequences', 'p_value', 'e_value']

	# Load the primary motif -- must resolve to exactly one entry.
	primary_names, primary_pwms = _load_motifs(primary_motif, 'primary_motif')
	if len(primary_names) != 1:
		raise ValueError("`primary_motif` must resolve to exactly one motif, "
			f"not {len(primary_names)}. If you have a dict of candidates, "
			"index into it yourself before calling `spamo`.")

	primary_name = primary_names[0]
	primary_pwm = primary_pwms[0]
	w_p = primary_pwm.shape[-1]

	# Load the secondary motifs.
	secondary_names, secondary_pwms = _load_motifs(secondary_motifs,
		'secondary_motifs')
	n_motifs = len(secondary_names)

	if n_motifs == 0:
		if n_jobs != -1:
			numba.set_num_threads(_n_jobs)
		return pandas.DataFrame(columns=columns)

	secondary_widths = numpy.array([pwm.shape[-1] for pwm in secondary_pwms],
		dtype=numpy.int64)

	# Build the log-odds PWMs and per-motif raw score thresholds, using the
	# same p-value-driven convention as fimo/centrimo.
	primary_fwd = numpy.log2(primary_pwm + eps) - math.log2(0.25)
	primary_rc = primary_fwd[::-1, ::-1].copy()

	primary_thresholds, _, _ = _pvalue_score_thresholds(primary_fwd,
		numpy.array([0, w_p], dtype=numpy.int64), bin_size, threshold)
	primary_threshold = primary_thresholds[0]

	fwd_lengths = numpy.cumsum([0] + list(secondary_widths)).astype(numpy.int64)
	fwd_concat = numpy.concatenate(secondary_pwms, axis=-1)

	if reverse_complement:
		rc_pwms = [pwm[::-1, ::-1] for pwm in secondary_pwms]
		all_concat = numpy.concatenate([fwd_concat] + rc_pwms, axis=-1)
		rc_lengths = fwd_lengths[1:] + fwd_lengths[-1]
		pwm_lengths = numpy.concatenate([fwd_lengths, rc_lengths]).astype(numpy.int64)
	else:
		all_concat = fwd_concat
		pwm_lengths = fwd_lengths

	log_pwm = numpy.log2(all_concat + eps) - math.log2(0.25)

	secondary_thresholds, _, _ = _pvalue_score_thresholds(
		log_pwm[:, :fwd_lengths[-1]], fwd_lengths, bin_size, threshold)

	# Load the sequences, requiring that they all have the same length.
	X, n_seqs, seq_len = _load_sequences(sequences, alphabet, seqlen)

	if n_seqs == 0:
		raise ValueError("Cannot run spamo with zero sequences.")

	if seq_len - 2 * margin < w_p:
		raise ValueError(f"Primary motif '{primary_name}' (width {w_p}) does "
			f"not fit within the central region after excluding `margin` "
			f"({margin}) from each edge of a sequence of length {seq_len}.")

	r = (2 * margin - w_p - secondary_widths) / 2.0
	if numpy.any(r <= 0):
		bad = secondary_names[int(numpy.argmin(r))]
		raise ValueError(f"Secondary motif '{bad}' does not fit within the "
			f"flanking `margin` ({margin}) region around the primary motif "
			f"'{primary_name}' (width {w_p}).")

	# Scan for the primary motif's best site per sequence.
	primary_positions, primary_strands = _spamo_primary_sites(X, n_seqs,
		seq_len, primary_fwd, primary_rc, w_p, primary_threshold, margin,
		reverse_complement)

	if (primary_positions == -1).all():
		if n_jobs != -1:
			numba.set_num_threads(_n_jobs)
		raise ValueError("No sequence has a primary motif site scoring above "
			"`threshold` within the central scan region.")

	# Scan for each secondary motif's best site per sequence, within `margin`
	# bp of the primary site.
	secondary_positions, secondary_strands = _spamo_secondary_sites(X, n_seqs,
		seq_len, primary_positions, primary_strands, w_p, log_pwm,
		pwm_lengths, secondary_thresholds, margin, reverse_complement,
		n_motifs)

	if n_jobs != -1:
		numba.set_num_threads(_n_jobs)

	# Compute the enrichment statistics for each secondary motif and strand
	# category. This operates on the small position/strand arrays only, and
	# so is not a throughput bottleneck.
	num_bins_per_side = int(numpy.ceil(range_ / bin_size_bp))
	n_bins_tested = 2 * num_bins_per_side

	rows = []
	for k in range(n_motifs):
		w_s = int(secondary_widths[k])
		r_k = (2 * margin - w_p - w_s) / 2.0
		q = bin_size_bp / (2 * r_k + 1)

		valid = (primary_positions != -1) & (secondary_positions[k] != -1)
		p_pos = primary_positions[valid]
		s_pos = secondary_positions[k, valid]
		same_strand = secondary_strands[k, valid] == primary_strands[valid]

		# Signed offset f: D = gap between closest edges; f = -(D+1) if the
		# secondary is 5' of the primary, else f = D+1 (f is never 0).
		secondary_first = (s_pos + w_s) <= p_pos
		D = numpy.where(secondary_first, p_pos - (s_pos + w_s),
			s_pos - (p_pos + w_p))
		sign = numpy.where(secondary_first, -1, 1)

		for category, mask in (('same', same_strand), ('opposite', ~same_strand)):
			n = int(mask.sum())

			if n == 0:
				rows.append((secondary_names[k], k, w_s, category, 0,
					n_bins_tested, numpy.nan, numpy.nan, 0, 1.0, 1.0))
				continue

			Dc, signc = D[mask], sign[mask]
			in_range = Dc < range_

			pos_bins = (Dc[in_range & (signc > 0)] // bin_size_bp).astype(numpy.int64)
			neg_bins = (Dc[in_range & (signc < 0)] // bin_size_bp).astype(numpy.int64)

			pos_counts = numpy.bincount(pos_bins, minlength=num_bins_per_side)[:num_bins_per_side]
			neg_counts = numpy.bincount(neg_bins, minlength=num_bins_per_side)[:num_bins_per_side]

			# q is identical for every bin (fixed width, fixed N), so
			# binom.sf is monotonic in the count -- the best bin is simply
			# whichever has the most sequences, no per-bin test needed.
			all_counts = numpy.concatenate([neg_counts, pos_counts])
			best = int(numpy.argmax(all_counts))
			best_count = int(all_counts[best])

			p_value = float(scipy.stats.binom.sf(best_count - 1, n, q))
			e_value = min(p_value * n_bins_tested * 2 * n_motifs, 1.0)

			if best < num_bins_per_side:
				b = best
				lo_D, hi_D = b * bin_size_bp, (b + 1) * bin_size_bp - 1
				best_offset_lo, best_offset_hi = -(hi_D + 1), -(lo_D + 1)
			else:
				b = best - num_bins_per_side
				lo_D, hi_D = b * bin_size_bp, (b + 1) * bin_size_bp - 1
				best_offset_lo, best_offset_hi = lo_D + 1, hi_D + 1

			rows.append((secondary_names[k], k, w_s, category, n,
				n_bins_tested, int(best_offset_lo), int(best_offset_hi),
				best_count, p_value, float(e_value)))

	result = pandas.DataFrame(rows, columns=columns)

	if return_site_positions:
		return (result, primary_positions, primary_strands,
			secondary_positions, secondary_strands)

	return result
