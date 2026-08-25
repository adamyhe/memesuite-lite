# test_spamo.py
# Contact: Adam He <adamyhe@gmail.com>

import numpy
import pandas
import pytest

from memelite.spamo import spamo
from memelite.spamo import _spamo_primary_sites
from memelite.spamo import _spamo_secondary_sites
from memelite.io import read_meme
from memelite.utils import one_hot_encode

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


COLUMNS = ('motif_name', 'motif_idx', 'width', 'orientation', 'n_sequences',
	'n_bins_tested', 'gap_lo', 'gap_hi', 'n_matching_sequences', 'p_value',
	'adj_p_value', 'e_value')


def _make_one_hot(shape, random_state=None):
	"""Build a correctly-formed random one-hot array with a fixed random
	state, all sequences the same length as required by SpaMo."""

	random_state = numpy.random.RandomState(random_state)

	n, c, l = shape
	idxs = random_state.randint(0, c, size=(n, l))

	ohe = numpy.zeros(shape, dtype='int8')
	for i in range(n):
		ohe[i, idxs[i], numpy.arange(l)] = 1

	return ohe


def _reverse_complement_str(seq, complement={'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}):
	return ''.join(complement[c] for c in reversed(seq))


def _plant(X, motif, offsets):
	"""Overwrite each sequence in `X` with `motif` at its given offset."""

	w = motif.shape[-1]
	for i, offset in enumerate(offsets):
		X[i, :, offset:offset+w] = motif
	return X


def _row(result, name, orientation):
	return result[(result['motif_name'] == name) &
		(result['orientation'] == orientation)].iloc[0]


def _has_row(result, name, orientation):
	return ((result['motif_name'] == name) &
		(result['orientation'] == orientation)).any()


###


def test_spamo_recovers_known_offset_downstream_same_strand():
	n_seqs, seq_len = 300, 500
	primary = one_hot_encode("ACGTGCA")
	secondary = one_hot_encode("TTGCCAA")
	w_p = primary.shape[-1]

	primary_pos = 246
	gap = 20
	secondary_pos = primary_pos + w_p + gap

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=0)
	_plant(X, primary, [primary_pos] * n_seqs)
	_plant(X, secondary, [secondary_pos] * 150)

	result = spamo({'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, threshold=0.001)

	assert tuple(result.columns) == COLUMNS

	row = _row(result, 's1', 'downstream_same')
	assert row['n_matching_sequences'] >= 0.9 * 150
	assert row['gap_lo'] <= gap <= row['gap_hi']
	assert row['e_value'] < 1e-10

	# Only quadrant 1 (downstream, same strand) has any signal, so the
	# categories built purely from other quadrants must not appear.
	assert not _has_row(result, 's1', 'upstream_same')
	assert not _has_row(result, 's1', 'upstream_opposite')
	assert not _has_row(result, 's1', 'downstream_opposite')


def test_spamo_recovers_known_offset_upstream():
	n_seqs, seq_len = 300, 500
	primary = one_hot_encode("ACGTGCA")
	secondary = one_hot_encode("TTGCCAA")
	w_s = secondary.shape[-1]

	primary_pos = 246
	gap = 15
	secondary_pos = primary_pos - gap - w_s

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=1)
	_plant(X, primary, [primary_pos] * n_seqs)
	_plant(X, secondary, [secondary_pos] * 150)

	result = spamo({'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, threshold=0.001)

	row = _row(result, 's1', 'upstream_same')
	assert row['n_matching_sequences'] >= 0.9 * 150
	assert row['gap_lo'] <= gap <= row['gap_hi']
	assert row['e_value'] < 1e-10
	assert not _has_row(result, 's1', 'downstream_same')


def test_spamo_same_vs_opposite_strand_discrimination():
	n_seqs, seq_len = 300, 500
	primary = one_hot_encode("ACGTGCA")
	secondary_seq = "TTGCCAA"
	secondary = one_hot_encode(secondary_seq)
	rc_secondary = one_hot_encode(_reverse_complement_str(secondary_seq))
	w_p = primary.shape[-1]

	primary_pos = 246
	gap = 20
	secondary_pos = primary_pos + w_p + gap

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=2)
	_plant(X, primary, [primary_pos] * n_seqs)
	_plant(X, rc_secondary, [secondary_pos] * 150)

	result = spamo({'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, threshold=0.001)

	row = _row(result, 's1', 'downstream_opposite')
	assert row['n_matching_sequences'] >= 0.9 * 150
	assert row['e_value'] < 1e-10
	assert not _has_row(result, 's1', 'downstream_same')


def test_spamo_side_rotates_with_primary_strand():
	# The primary is planted on the reverse-complement strand. A secondary
	# site placed literally *before* the primary in raw sequence
	# coordinates should be reported as "downstream" once rotated into the
	# primary's own (reverse) reading direction, not "upstream" -- this is
	# the side-rotation behavior confirmed against `bin_matches` in the
	# reference binary's C source (src/spamo-matches.c).
	n_seqs, seq_len = 300, 500
	primary_seq, secondary_seq = "ACGTGCA", "TTGCCAA"
	primary = one_hot_encode(primary_seq)
	rc_primary = one_hot_encode(_reverse_complement_str(primary_seq))
	secondary = one_hot_encode(secondary_seq)
	rc_secondary = one_hot_encode(_reverse_complement_str(secondary_seq))
	w_s = secondary.shape[-1]

	primary_pos = 246
	gap = 20
	secondary_pos = primary_pos - gap - w_s

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=40)
	_plant(X, rc_primary, [primary_pos] * n_seqs)
	_plant(X, rc_secondary, [secondary_pos] * 150)

	result = spamo({'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, threshold=0.001)

	row = _row(result, 's1', 'downstream_same')
	assert row['n_matching_sequences'] >= 0.9 * 150
	assert row['gap_lo'] <= gap <= row['gap_hi']
	assert not _has_row(result, 's1', 'upstream_same')


def test_spamo_pooled_secondary_pal_orientation():
	# Half the sequences have the secondary matching the forward strand
	# downstream of the primary; the other half have its reverse complement
	# at the same gap and side. Neither raw quadrant alone has the full
	# signal, but `downstream_secondary_pal` (which pools both strands on
	# the downstream side) should recover it clearly.
	n_seqs, seq_len = 400, 500
	primary = one_hot_encode("ACGTGCA")
	secondary_seq = "TTGCCAA"
	secondary = one_hot_encode(secondary_seq)
	rc_secondary = one_hot_encode(_reverse_complement_str(secondary_seq))
	w_p = primary.shape[-1]

	primary_pos = 246
	gap = 20
	secondary_pos = primary_pos + w_p + gap

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=42)
	_plant(X, primary, [primary_pos] * n_seqs)
	# Plant the forward copy of the secondary in sequences [0, 100) and the
	# rc copy in a disjoint block [100, 200), both at the same gap/side.
	_plant(X[:100], secondary, [secondary_pos] * 100)
	_plant(X[100:200], rc_secondary, [secondary_pos] * 100)

	result = spamo({'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, threshold=0.001)

	pooled = _row(result, 's1', 'downstream_secondary_pal')
	assert pooled['n_matching_sequences'] >= 190
	assert pooled['e_value'] < 1e-10


def test_spamo_leftover_bin_boundaries():
	n_seqs, seq_len = 300, 200
	primary = one_hot_encode("ACGTGCA")
	secondary = one_hot_encode("TTGCCAA")
	w_p, w_s = primary.shape[-1], secondary.shape[-1]

	margin, bin_size_bp = 25, 5
	# quad_opt_count = margin - w_s + 1 = 19; 19 % 5 = 4, so the leftover
	# bin covers gaps [15, 18] -- narrower than a full bin_size_bp=5 bin.
	primary_pos = 90
	gap = 17
	secondary_pos = primary_pos + w_p + gap

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=41)
	_plant(X, primary, [primary_pos] * n_seqs)
	_plant(X, secondary, [secondary_pos] * 200)

	result = spamo({'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, margin=margin,
		range_=margin, bin_size_bp=bin_size_bp, threshold=0.001)

	row = _row(result, 's1', 'downstream_same')
	assert row['gap_lo'] == 15
	assert row['gap_hi'] == 18


def test_spamo_no_signal_omits_secondary_motif():
	n_seqs, seq_len = 300, 500
	primary = one_hot_encode("ACGTGCA")
	secondary = one_hot_encode("TTGCCAA")
	w_p, w_s = primary.shape[-1], secondary.shape[-1]

	r = numpy.random.RandomState(3)
	X = _make_one_hot((n_seqs, 4, seq_len), random_state=3)
	_plant(X, primary, [246] * n_seqs)

	offsets = r.randint(246 + w_p, seq_len - w_s, size=150)
	_plant(X, secondary, offsets)

	# The default `evalue_threshold` (10.0, matching the reference binary's
	# own generous default) is loose enough that even random noise's best
	# bin usually still gets reported as a fallback -- use a strict
	# `evalue_threshold` to directly exercise the "omitted entirely" path.
	result = spamo({'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, threshold=0.001,
		evalue_threshold=1e-6)

	assert result[result['motif_name'] == 's1'].empty


def test_spamo_no_signal_reports_fallback_row_at_default_evalue():
	n_seqs, seq_len = 300, 500
	primary = one_hot_encode("ACGTGCA")
	secondary = one_hot_encode("TTGCCAA")
	w_p, w_s = primary.shape[-1], secondary.shape[-1]

	r = numpy.random.RandomState(3)
	X = _make_one_hot((n_seqs, 4, seq_len), random_state=3)
	_plant(X, primary, [246] * n_seqs)

	offsets = r.randint(246 + w_p, seq_len - w_s, size=150)
	_plant(X, secondary, offsets)

	result = spamo({'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, threshold=0.001)

	# At the default (loose) evalue_threshold, exactly one fallback row is
	# reported, and it is not independently significant (its own
	# adj_p_value is above `cutoff` -- that's precisely why it's a
	# fallback rather than one of possibly-several significant rows).
	sub = result[result['motif_name'] == 's1']
	assert len(sub) == 1
	assert sub.iloc[0]['adj_p_value'] > 0.05


def test_spamo_reverse_complement_primary():
	n_seqs, seq_len = 300, 500
	primary_seq = "ACGTGCA"
	primary = one_hot_encode(primary_seq)
	rc_primary = one_hot_encode(_reverse_complement_str(primary_seq))
	secondary = one_hot_encode("TTGCCAA")
	w_p = primary.shape[-1]

	primary_pos = 246
	gap = 20
	secondary_pos = primary_pos + w_p + gap

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=4)
	_plant(X, rc_primary, [primary_pos] * n_seqs)
	_plant(X, secondary, [secondary_pos] * 150)

	result, p_pos, p_strand, s_pos, s_strand = spamo(
		{'p1': primary.astype('float64')}, {'s1': secondary.astype('float64')},
		X, threshold=0.001, return_site_positions=True)

	found = p_pos != -1
	assert found.sum() >= 0.9 * n_seqs
	# A handful of sequences may tie with a coincidental exact background
	# match elsewhere and break toward the (also correct, by construction)
	# forward strand, so allow a small fraction of non-rc calls.
	assert (p_strand[found] == 1).mean() >= 0.95


##


def test_spamo_primary_must_resolve_to_one_motif():
	motifs = read_meme("tests/data/test.meme")
	X = _make_one_hot((10, 4, 500), random_state=5)

	assert_raises(ValueError, spamo, motifs, motifs, X)


def test_spamo_no_qualifying_primary_sequences_raises():
	primary = one_hot_encode("ACGTACGTAC")
	secondary = one_hot_encode("TTGCCAA")
	X = _make_one_hot((10, 4, 500), random_state=6)

	assert_raises(ValueError, spamo, {'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, threshold=1e-12)


def test_spamo_primary_wider_than_scan_region_raises():
	primary = one_hot_encode("ACGTACGTAC")  # width 10
	secondary = one_hot_encode("TTGCCAA")
	X = _make_one_hot((10, 4, 50), random_state=7)

	assert_raises(ValueError, spamo, {'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, margin=30)


def test_spamo_secondary_wider_than_margin_raises():
	primary = one_hot_encode("ACGTGCA")  # width 7
	secondary = one_hot_encode("A" * 20)  # width 20
	X = _make_one_hot((10, 4, 500), random_state=8)

	assert_raises(ValueError, spamo, {'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, X, margin=10)


def test_spamo_no_secondary_motifs():
	primary = one_hot_encode("ACGTGCA")
	X = _make_one_hot((10, 4, 500), random_state=9)

	result = spamo({'p1': primary.astype('float64')}, {}, X)

	assert isinstance(result, pandas.DataFrame)
	assert result.empty
	assert tuple(result.columns) == COLUMNS


def test_spamo_fasta_seqlen_filter(tmp_path):
	r = numpy.random.RandomState(20)
	primary_seq, secondary_seq = "ACGTGCA", "TTGCCAA"
	seq_len = 500
	primary_pos = 246
	secondary_pos = primary_pos + len(primary_seq) + 20

	def make_seq():
		bases = list(r.choice(list("ACGT"), size=seq_len))
		bases[primary_pos:primary_pos+len(primary_seq)] = list(primary_seq)
		bases[secondary_pos:secondary_pos+len(secondary_seq)] = list(secondary_seq)
		return ''.join(bases)

	seq1, seq2 = make_seq(), make_seq()
	seq3 = ''.join(r.choice(list("ACGT"), size=100))

	fasta_path = tmp_path / "seqs.fa"
	fasta_path.write_text(f">seq1\n{seq1}\n>seq2\n{seq2}\n>seq3\n{seq3}\n")

	primary = one_hot_encode(primary_seq)
	secondary = one_hot_encode(secondary_seq)

	result = spamo({'p1': primary.astype('float64')},
		{'s1': secondary.astype('float64')}, str(fasta_path), threshold=0.001)

	sub = result[result['motif_name'] == 's1']
	assert not sub.empty
	assert (sub['n_sequences'] <= 2).all()
	assert (sub['n_matching_sequences'] >= 1).all()


def test_spamo_return_site_positions():
	n_seqs, seq_len = 10, 500
	primary = one_hot_encode("ACGTGCA")
	secondary = one_hot_encode("TTGCCAA")

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=10)
	_plant(X, primary, [246] * n_seqs)

	result, p_pos, p_strand, s_pos, s_strand = spamo(
		{'p1': primary.astype('float64')}, {'s1': secondary.astype('float64')},
		X, return_site_positions=True)

	assert p_pos.shape == (n_seqs,)
	assert p_strand.shape == (n_seqs,)
	assert s_pos.shape == (1, n_seqs)
	assert s_strand.shape == (1, n_seqs)


def test_spamo_meme_file_input():
	motifs = read_meme("tests/data/test.meme")
	names = list(motifs.keys())
	primary = {names[0]: motifs[names[0]]}
	secondary = {n: motifs[n] for n in names[1:]}

	X = _make_one_hot((50, 4, 500), random_state=11)

	result = spamo(primary, secondary, X, threshold=0.01)

	assert tuple(result.columns) == COLUMNS
	assert (result['p_value'] >= 0).all()
	assert (result['p_value'] <= 1).all()
	assert (result['adj_p_value'] >= result['p_value']).all()


def test_spamo_evalue_formula():
	n_seqs, seq_len = 300, 500
	primary = one_hot_encode("ACGTGCA")
	secondary_pwms = {
		's1': one_hot_encode("TTGCCAA").astype('float64'),
		's2': one_hot_encode("GGCATTA").astype('float64'),
	}
	w_p = primary.shape[-1]

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=12)
	_plant(X, primary, [246] * n_seqs)
	_plant(X, one_hot_encode("TTGCCAA"), [246 + w_p + 20] * 150)

	result = spamo({'p1': primary.astype('float64')}, secondary_pwms, X,
		threshold=0.001)

	# `e_value` is a motif-level quantity: constant across every row of a
	# given secondary motif, and equal to the *smallest* adj_p_value among
	# its reported rows times the number of secondary motifs (the globally
	# best (orientation, bin) is always one of the reported rows whenever
	# any row is reported at all, since reporting requires adj_p_value <=
	# cutoff, and the global minimum is <= any value that already cleared
	# cutoff).
	n_motifs = len(secondary_pwms)
	for name in result['motif_name'].unique():
		sub = result[result['motif_name'] == name]
		assert (sub['e_value'] == sub['e_value'].iloc[0]).all()
		expected = sub['adj_p_value'].min() * n_motifs
		assert abs(sub['e_value'].iloc[0] - expected) < 1e-9


##


def test_spamo_primary_tie_break_leftmost_and_forward():
	# A single-base primary motif that strongly prefers 'A' over the other
	# three bases, planted (as an 'A') at two positions in an otherwise
	# all-'C' sequence, both forward and reverse-complement scoring equally
	# at each -- the tie-break should deterministically pick the leftmost
	# position, on the forward strand.
	seq_len = 30
	margin = 5
	X = numpy.full(seq_len, 1, dtype=numpy.int8)  # all 'C'
	X[10] = 0  # 'A'
	X[20] = 0  # 'A'

	pwm_fwd = numpy.array([[5.0], [0.0], [0.0], [0.0]])
	pwm_rc = pwm_fwd.copy()  # a single-base motif is its own reverse complement here

	positions, strands = _spamo_primary_sites(X, 1, seq_len, pwm_fwd, pwm_rc,
		1, -100.0, margin, True)

	assert positions[0] == 10
	assert strands[0] == 0


def test_spamo_primary_margin_boundary_exclusion():
	seq_len = 30
	margin = 5
	# A site exactly at the excluded edge (position margin - 1) must not be
	# found; a site one bp further in (position margin) must be.
	X = numpy.full(seq_len, 1, dtype=numpy.int8)
	X[margin - 1] = 0
	X[margin] = 0

	pwm_fwd = numpy.array([[5.0], [0.0], [0.0], [0.0]])
	pwm_rc = pwm_fwd.copy()

	positions, strands = _spamo_primary_sites(X, 1, seq_len, pwm_fwd, pwm_rc,
		1, -100.0, margin, False)

	assert positions[0] == margin


def test_spamo_secondary_overlap_exclusion():
	# The primary site occupies [10, 11). A secondary candidate overlapping
	# it (position 10) must be skipped in favor of the next-best
	# non-overlapping site (position 12), even though the overlapping site
	# would otherwise score just as well.
	seq_len = 30
	margin = 10
	X = numpy.full(seq_len, 1, dtype=numpy.int8)
	X[10] = 0  # inside the primary's own span
	X[12] = 0  # a valid, non-overlapping secondary site

	primary_positions = numpy.array([10], dtype=numpy.int32)
	primary_strands = numpy.array([0], dtype=numpy.int8)

	pwm = numpy.array([[5.0], [0.0], [0.0], [0.0]])
	pwm_lengths = numpy.array([0, 1], dtype=numpy.int64)
	score_thresholds = numpy.array([-100.0])

	positions, strands = _spamo_secondary_sites(X, 1, seq_len,
		primary_positions, primary_strands, 1, pwm, pwm_lengths,
		score_thresholds, margin, False, 1)

	assert positions[0, 0] == 12


def test_spamo_secondary_window_boundary_exclusion():
	# A secondary site just beyond `margin` bp from the primary must be
	# excluded; one within `margin` must be found.
	seq_len = 40
	margin = 10
	X = numpy.full(seq_len, 1, dtype=numpy.int8)
	primary_pos = 15
	X[primary_pos + 1 + margin] = 0       # just outside the window
	X[primary_pos + 1 + margin - 1] = 0   # just inside the window

	primary_positions = numpy.array([primary_pos], dtype=numpy.int32)
	primary_strands = numpy.array([0], dtype=numpy.int8)

	pwm = numpy.array([[5.0], [0.0], [0.0], [0.0]])
	pwm_lengths = numpy.array([0, 1], dtype=numpy.int64)
	score_thresholds = numpy.array([-100.0])

	positions, strands = _spamo_secondary_sites(X, 1, seq_len,
		primary_positions, primary_strands, 1, pwm, pwm_lengths,
		score_thresholds, margin, False, 1)

	assert positions[0, 0] == primary_pos + 1 + margin - 1
