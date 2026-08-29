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


# The 9 "biologically interesting" orientation categories from the reference
# SpaMo binary (src/spamo-matches.c): the 4 raw quadrants (upstream/
# downstream x same/opposite strand, relative to the primary's own matched
# strand) plus 5 combinations that pool pairs of quadrants together to stay
# significant even if a motif's actual strand identity doesn't matter
# biologically (e.g. because it's palindromic). The raw-quadrant index
# order here (quad = 2*side + strand, side the high bit) and the pairing of
# quadrants into each combined category were confirmed empirically against
# the real binary (v5.5.9) -- the source's own `#define`/arithmetic reads as
# the opposite bit order, but doesn't reproduce the binary's actual reported
# orientation for a given planted signal; a set of controlled probes (each
# varying exactly one of side/strand/primary-strand) resolved the true
# mapping unambiguously. All 9 names are used regardless of
# `reverse_complement` -- confirmed empirically that the real binary's
# `-norc` does *not* collapse to a simpler category set (see `spamo()`'s
# docstring/implementation for why).
_ORIENTATION_NAMES_RC = (
	'upstream_same', 'upstream_opposite', 'downstream_same', 'downstream_opposite',
	'upstream_secondary_pal', 'upstream_primary_pal', 'downstream_primary_pal',
	'downstream_secondary_pal', 'both_pal',
)


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
	alphabet=('A', 'C', 'G', 'T'), margin=150, range_=150, bin_size_bp=1,
	bin_size=0.1, eps=0.0001, threshold=0.001, cutoff=0.05,
	evalue_threshold=10.0, reverse_complement=True, seqlen=None,
	return_site_positions=False, n_jobs=-1):
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

	Algorithm (verified against the real MEME suite v5.5.9 C source,
	`src/spamo-matches.c`/`src/spamo.c`, not just the docs/paper): for each
	sequence, the primary motif's single best site is found within the
	*central* region only (excluding `margin` bp from each edge, so a full
	flanking window always fits around wherever it lands). For each
	sequence with a qualifying primary site, each secondary motif's single
	best site is then found within `margin` bp on either side of the
	primary site, excluding the primary's own span (the two sites can never
	overlap). Each such pair is classified into one of 4 base quadrants --
	upstream/downstream (relative to the *primary's own matched strand*,
	i.e. rotated 180 degrees when the primary matched on the reverse
	strand) crossed with same/opposite strand -- and its gap `D` (the
	number of bp between the two sites' closest edges, counting from 0) is
	binned into `bin_size_bp`-wide bins ranging over `[0, margin - w_s)`
	(`w_s` = the secondary motif's width).

	Each of the 4 quadrants is tested independently at every bin, plus 5
	further categories that pool pairs of quadrants together so that a
	spacing constraint involving a motif that behaves symmetrically with
	respect to strand (e.g. because it's palindromic, or simply because the
	orientation doesn't matter biologically) isn't missed by requiring a
	specific strand match: `upstream_secondary_pal`/`downstream_secondary_pal`
	(pool both strands on one side), `upstream_primary_pal`/
	`downstream_primary_pal` (pool the two "diagonal" quadrant pairs), and
	`both_pal` (all 4 quadrants). All 9 categories share **one** binomial
	test denominator `N` (every sequence with a qualifying primary+secondary
	pair, regardless of which quadrant it landed in) and a null probability
	proportional to how much of the total 4-quadrant space that bin/category
	covers -- not two independent same-strand/opposite-strand tests with
	separate `N`s, which is what this package's implementation did before
	this was corrected against source. All 9 categories, and the same
	4-quadrant-space denominator, are used regardless of
	`reverse_complement`: with it False, only the forward strand is ever
	scanned, so the opposite-strand quadrants (and any combined category
	built from one) are always empty -- confirmed empirically that the real
	binary's `-norc` behaves the same way, rather than collapsing to a
	simpler 2-category model. Every `(orientation, bin)` p-value is
	Bonferroni-corrected (`adj_p_value`) by 9 times the number of bins
	tested (bounded by `range_`); a further, *separate* correction by the
	number of secondary motifs gives the motif-level `e_value`.

	A secondary motif's rows are: every `(orientation, bin)` whose
	`adj_p_value` is at or below `cutoff`; or, if none qualify but the
	motif's own `e_value` is at or below `evalue_threshold`, just the single
	best `(orientation, bin)` as a fallback; or no rows at all if neither
	holds -- so, unlike most other functions in this package, secondary
	motifs found to be non-enriched are omitted from the output entirely
	rather than reported as an all-zero row (matching the reference
	binary's own behavior exactly).

	Only the best secondary site per sequence is used (matching the
	reference SpaMo binary's `-usebestsec` option, but as the *only*
	behavior here rather than a non-default flag): the underlying binomial
	test assumes exactly one gap value per sequence per quadrant, which is
	only statistically valid under best-site semantics -- the reference
	binary's default "count all matches above threshold" mode would let one
	sequence contribute multiple gaps, violating that assumption. This
	mirrors `centrimo`'s identical, previously-made choice in this package.

	Not implemented: `-trim` (motif-edge information-content trimming --
	apply it to your PWMs yourself before calling `spamo`, if wanted),
	`-shared`/`-overlap`/`-joint` (secondary-motif redundancy clustering),
	multiple named secondary-motif databases (pass one flat dict),
	`-dumpseqs`, and `-keepprimary`'s erasure of "extra" occurrences of the
	primary motif elsewhere in a sequence (the reference binary does this by
	default; for equivalence testing, pass `-keepprimary` to the reference
	binary to disable it instead). As in `fimo`/`centrimo`, `threshold` is a
	p-value (applied to both the primary and secondary motifs), converted
	internally to a raw score threshold, rather than the reference binary's
	fixed-bits `-minscore`.


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

	range_: int, optional
		How far out (bp, on either side of the primary site) the candidate
		gap bins for the significance search extend. Sequences with a gap
		beyond `range_` (but still within `margin`) still count toward a
		motif's `n_sequences` and contribute to a bin's count, but are not
		considered when searching for the most significant bin. Deliberately
		*not* tied to `margin` by default (matching the reference binary,
		which has an independent, fixed default here rather than deriving
		it from `-margin`). Default is 150.

	bin_size_bp: int, optional
		The width, in bp, of each candidate gap bin. Default is 1.

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

	cutoff: float, optional
		The (Bonferroni-corrected) p-value a single `(orientation, bin)`
		must reach to be reported as its own row, mirroring the reference
		binary's `-cutoff`. Default is 0.05.

	evalue_threshold: float, optional
		The motif-level E-value a secondary motif's best `(orientation,
		bin)` must reach for that motif to be reported at all (as a
		fallback single row, if no individual bin/orientation reached
		`cutoff`), mirroring the reference binary's `-evalue`. Default is
		10.0.

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
		A dataframe with 0 to 9 rows per secondary motif (see above for
		when a motif is omitted, and when it gets multiple rows), columns
		`motif_name, motif_idx, width, orientation, n_sequences,
		n_bins_tested, gap_lo, gap_hi, n_matching_sequences, p_value,
		adj_p_value, e_value`. `orientation` is one of the 9 category names
		described above (only the 4 `same`-strand-including ones can ever
		appear when `reverse_complement` is False).
		`gap_lo`/`gap_hi` are the inclusive bp range (edge-to-edge, always
		non-negative) of the reported bin. `e_value` is a motif-level
		quantity, identical across every row belonging to the same
		secondary motif.

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

	columns = ['motif_name', 'motif_idx', 'width', 'orientation', 'n_sequences',
		'n_bins_tested', 'gap_lo', 'gap_hi', 'n_matching_sequences', 'p_value',
		'adj_p_value', 'e_value']

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

	quad_opt_counts = margin - secondary_widths + 1
	if numpy.any(quad_opt_counts <= 0):
		bad = secondary_names[int(numpy.argmin(quad_opt_counts))]
		raise ValueError(f"Secondary motif '{bad}' (width "
			f"{int(secondary_widths[int(numpy.argmin(quad_opt_counts))])}) "
			f"does not fit within the flanking `margin` ({margin}) region.")

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

	# Compute the enrichment statistics for each secondary motif. This
	# operates on the small position/strand arrays only, and so is not a
	# throughput bottleneck.
	#
	# The full 9-orientation model (and its 4x-quad_opt_count null-
	# probability denominator) is used regardless of `reverse_complement`
	# -- confirmed empirically against the real binary's `-norc` mode
	# (which does *not* collapse to a simpler 2-orientation/2x-denominator
	# model, despite the source's own `revcomp ? 4 : 2`-style branches
	# reading as if it should; those apparently key off whether the
	# alphabet itself is complementable, e.g. DNA, not the `-norc` flag).
	# `same_strand` is trivially always True when `reverse_complement` is
	# False (both strand values are always 0), so signal only ever lands in
	# the even-indexed raw quadrants (0=upstream, 2=downstream) and the
	# categories built from odd-indexed ones are always empty -- exactly
	# matching what the real binary's own `-norc` output looks like.
	orientation_names = _ORIENTATION_NAMES_RC
	mult = numpy.array([1, 1, 1, 1, 2, 2, 2, 2, 4])
	test_max = int(numpy.ceil(range_ / bin_size_bp))

	rows = []
	for k in range(n_motifs):
		w_s = int(secondary_widths[k])

		quad_opt_count = margin - w_s + 1
		quad_bin_count = quad_opt_count // bin_size_bp
		quad_leftover = quad_opt_count % bin_size_bp
		n_bins = quad_bin_count + (1 if quad_leftover else 0)

		valid = (primary_positions != -1) & (secondary_positions[k] != -1)
		n_total = int(valid.sum())

		if n_total == 0 or n_bins == 0:
			continue

		p_pos = primary_positions[valid]
		p_strand = primary_strands[valid]
		s_pos = secondary_positions[k, valid]
		s_strand = secondary_strands[k, valid]

		# Gap D (edge-to-edge, counting from 0) between the primary and
		# secondary sites -- unaffected by strand, since it's a purely
		# positional quantity.
		secondary_first = (s_pos + w_s) <= p_pos
		D = numpy.where(secondary_first, p_pos - (s_pos + w_s),
			s_pos - (p_pos + w_p))

		# "Upstream"/"downstream" is relative to the *primary's own*
		# matched strand, not literal sequence coordinates: rotate the
		# side label when the primary matched on the reverse strand.
		# Verified against `bin_matches` in the reference binary's C
		# source (`src/spamo-matches.c`).
		side = numpy.where(secondary_first ^ (p_strand == 1), 0, 1)
		same_strand = s_strand == p_strand
		opposite_bit = (~same_strand).astype(numpy.int64)
		quad = 2 * side + opposite_bit
		# quad 0 = upstream_same, 1 = upstream_opposite,
		# 2 = downstream_same, 3 = downstream_opposite.

		bin_idx = numpy.minimum(D // bin_size_bp, n_bins - 1)

		quad_counts = numpy.zeros((4, n_bins), dtype=numpy.int64)
		for q in range(4):
			sel = quad == q
			if sel.any():
				quad_counts[q] = numpy.bincount(bin_idx[sel], minlength=n_bins)

		# The 5 "palindrome-tolerant" combinations -- each pools a pair (or
		# all 4) of the raw quadrant counts above.
		counts = numpy.empty((9, n_bins), dtype=numpy.int64)
		counts[0:4] = quad_counts
		counts[4] = quad_counts[0] + quad_counts[1]  # upstream_secondary_pal
		counts[5] = quad_counts[0] + quad_counts[3]  # upstream_primary_pal
		counts[6] = quad_counts[1] + quad_counts[2]  # downstream_primary_pal
		counts[7] = quad_counts[2] + quad_counts[3]  # downstream_secondary_pal
		counts[8] = counts[4] + counts[7]            # both_pal

		# Null probability for a bin: its share of the total space spanned
		# by all 4 quadrants (not just the quadrant(s) this orientation
		# covers), scaled by how many quadrants this orientation pools.
		denom = 4 * quad_opt_count
		base_probs = numpy.full(n_bins, bin_size_bp / denom)
		if quad_leftover:
			base_probs[-1] = quad_leftover / denom
		probs = mult[:, None] * base_probs[None, :]

		p_values = scipy.stats.binom.sf(counts - 1, n_total, probs)

		n_bins_tested = min(quad_bin_count, test_max) + (1 if quad_leftover else 0)
		tests = 9 * n_bins_tested
		adj_p_values = p_values * tests

		testable = numpy.arange(n_bins) < test_max
		if not testable.any():
			continue

		masked_adj = numpy.where(testable[None, :], adj_p_values, numpy.inf)
		best_orient, best_bin = numpy.unravel_index(
			numpy.argmin(masked_adj), masked_adj.shape)
		e_value = float(masked_adj[best_orient, best_bin]) * n_motifs

		sig_orients, sig_bins = numpy.where(
			testable[None, :] & (adj_p_values <= cutoff))

		if len(sig_orients) == 0:
			if e_value > evalue_threshold:
				continue
			sig_orients = numpy.array([best_orient])
			sig_bins = numpy.array([best_bin])

		for o, b in zip(sig_orients, sig_bins):
			gap_lo = int(b) * bin_size_bp
			gap_hi = gap_lo + bin_size_bp - 1
			if quad_leftover and b == quad_bin_count:
				gap_hi = quad_opt_count - 1

			rows.append((secondary_names[k], k, w_s, orientation_names[o],
				n_total, n_bins_tested, gap_lo, gap_hi, int(counts[o, b]),
				float(p_values[o, b]), float(adj_p_values[o, b]), e_value))

	result = pandas.DataFrame(rows, columns=columns)

	if return_site_positions:
		return (result, primary_positions, primary_strands,
			secondary_positions, secondary_strands)

	return result
