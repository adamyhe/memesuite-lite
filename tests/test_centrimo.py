# test_centrimo.py
# Contact: Adam He <adamyhe@gmail.com>

import numpy
import pandas
import pytest

from memelite.centrimo import centrimo
from memelite.centrimo import _centrimo_best_sites
from memelite.io import read_meme
from memelite.utils import one_hot_encode

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


COLUMNS = ('motif_name', 'motif_idx', 'width', 'n_sequences',
	'n_valid_positions', 'best_window_width', 'n_matching_sequences',
	'p_value', 'e_value')

CONTROL_COLUMNS = COLUMNS + ('n_control_sequences',
	'n_control_matching_sequences', 'fisher_p_value', 'fisher_e_value')


def _make_one_hot(shape, random_state=None):
	"""Build a correctly-formed random one-hot array with a fixed random
	state, all sequences the same length as required by CentriMo."""

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


###


def test_centrimo_recovers_central_enrichment():
	n_seqs, seq_len = 200, 101
	motif = one_hot_encode("ACGTACG")
	w = motif.shape[-1]
	center = (seq_len - w) // 2

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=0)
	n_planted = 120
	_plant(X, motif, [center] * n_planted)

	result = centrimo({'m1': motif.astype('float64')}, X, threshold=0.05)

	assert tuple(result.columns) == COLUMNS
	assert result.shape == (1, len(COLUMNS))

	row = result.iloc[0]
	assert row['n_matching_sequences'] >= 0.9 * n_planted
	assert row['best_window_width'] <= 41
	assert row['e_value'] < 1e-10


def test_centrimo_no_enrichment_uniform_random():
	n_seqs, seq_len = 200, 101
	motif = one_hot_encode("ACGTACG")
	w = motif.shape[-1]

	r = numpy.random.RandomState(1)
	X = _make_one_hot((n_seqs, 4, seq_len), random_state=1)
	n_planted = 120
	offsets = r.randint(0, seq_len - w + 1, size=n_planted)
	_plant(X, motif, offsets)

	result = centrimo({'m1': motif.astype('float64')}, X, threshold=0.05)

	row = result.iloc[0]
	assert row['e_value'] > 0.01


def test_centrimo_reverse_complement():
	n_seqs, seq_len = 200, 101
	seq = "ACGTACG"
	rc_seq = _reverse_complement_str(seq)

	motif = one_hot_encode(seq)
	rc_motif = one_hot_encode(rc_seq)
	w = motif.shape[-1]
	center = (seq_len - w) // 2

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=2)
	n_planted = 120
	_plant(X, rc_motif, [center] * n_planted)

	with_rc = centrimo({'m1': motif.astype('float64')}, X, threshold=0.05,
		reverse_complement=True)
	without_rc = centrimo({'m1': motif.astype('float64')}, X, threshold=0.05,
		reverse_complement=False)

	assert with_rc.iloc[0]['e_value'] < 1e-10
	assert (without_rc.iloc[0]['n_matching_sequences'] <
		with_rc.iloc[0]['n_matching_sequences'])


def test_centrimo_tie_averaging():
	# A single-base motif that strongly prefers 'A' (index 0) over the other
	# three bases, planted at two positions (2 and 3) in an otherwise all-'C'
	# sequence. Both positions tie for the best score, so the reported
	# distance should be the *average* of their centers (width 1, so centers
	# equal start positions: (2 + 3) / 2 = 2.5), not an arbitrary pick.
	X = numpy.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=numpy.int8)
	pwm = numpy.array([[5.0], [0.0], [0.0], [0.0]])
	pwm_lengths = numpy.array([0, 1], dtype=numpy.int64)
	score_thresholds = numpy.array([-100.0])

	distances, best_scores = _centrimo_best_sites(X, 1, 10, pwm, pwm_lengths,
		score_thresholds, False, 1)

	expected_center = (2 + 3) / 2
	expected_distance = expected_center - (10 - 1) / 2.0
	assert_array_almost_equal(distances, [[expected_distance]])
	assert_array_almost_equal(best_scores, [[5.0]])


def test_centrimo_below_threshold_excluded():
	seq_len = 20
	motif = one_hot_encode("ACGTACG")
	w = motif.shape[-1]

	X = _make_one_hot((2, 4, seq_len), random_state=3)
	_plant(X, motif, [5])

	result, distances = centrimo({'m1': motif.astype('float64')}, X,
		threshold=0.05, return_site_distances=True)

	assert result.iloc[0]['n_sequences'] == 1
	assert not numpy.isnan(distances[0, 0])
	assert numpy.isnan(distances[0, 1])


def test_centrimo_fasta_seqlen_filter(tmp_path):
	motif = one_hot_encode("ACGTACG")

	fasta_path = tmp_path / "seqs.fa"
	fasta_path.write_text(
		">seq1\n" + "ACGTACGT" * 10 + "\n"      # length 80
		">seq2\n" + "ACGTACGT" * 10 + "\n"      # length 80
		">seq3\n" + "ACGT" * 10 + "\n"           # length 40
	)

	result_default = centrimo({'m1': motif.astype('float64')}, str(fasta_path))
	assert result_default.iloc[0]['n_sequences'] <= 2

	result_short = centrimo({'m1': motif.astype('float64')}, str(fasta_path),
		seqlen=40)
	assert result_short.iloc[0]['n_sequences'] <= 1


def test_centrimo_no_motifs():
	X = _make_one_hot((5, 4, 20), random_state=4)
	result = centrimo({}, X)

	assert isinstance(result, pandas.DataFrame)
	assert result.empty
	assert tuple(result.columns) == COLUMNS


def test_centrimo_sequence_shorter_than_motif():
	motif = one_hot_encode("ACGTACGTAC")  # width 10
	X = _make_one_hot((5, 4, 5), random_state=5)

	assert_raises(ValueError, centrimo, {'m1': motif.astype('float64')}, X)


def test_centrimo_meme_file_input():
	motifs = read_meme("tests/data/test.meme")
	X = _make_one_hot((20, 4, 150), random_state=6)

	result = centrimo(motifs, X, threshold=0.05)

	assert result.shape == (12, len(COLUMNS))
	assert (result['p_value'] >= 0).all()
	assert (result['p_value'] <= 1).all()
	assert (result['e_value'] >= 0).all()


def test_centrimo_return_site_distances():
	motif = one_hot_encode("ACGTACG")
	X = _make_one_hot((10, 4, 50), random_state=7)

	result, distances = centrimo({'m1': motif.astype('float64')}, X,
		return_site_distances=True)

	assert distances.shape == (1, 10)


##


def test_centrimo_control_sequences_differential():
	n_seqs, seq_len = 200, 101
	motif = one_hot_encode("ACGTACG")
	w = motif.shape[-1]
	center = (seq_len - w) // 2

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=10)
	_plant(X, motif, [center] * 150)

	# A control set where the motif is present just as often, but always at
	# one of the two extreme ends rather than the center -- this is a much
	# stronger test than a motif-free control, since it confirms the test
	# distinguishes *where* matches fall rather than merely *whether* they
	# exist (with a strict, exact-match-only threshold, a plain random
	# background may by chance contribute zero qualifying control sites,
	# which only demonstrates the degenerate n_control_sequences == 0 case
	# covered by the threshold-exclusion design, not this comparison).
	r = numpy.random.RandomState(11)
	X_control = _make_one_hot((n_seqs, 4, seq_len), random_state=11)
	offsets = r.choice([0, seq_len - w], size=150)
	_plant(X_control, motif, offsets)

	result = centrimo({'m1': motif.astype('float64')}, X,
		control_sequences=X_control, threshold=0.05)

	assert tuple(result.columns) == CONTROL_COLUMNS

	row = result.iloc[0]
	assert row['n_control_sequences'] >= 100
	assert row['n_control_matching_sequences'] <= row['n_control_sequences']
	assert row['fisher_e_value'] < 1e-10


def test_centrimo_control_sequences_different_length():
	# The primary and control sets need not match in length or count.
	motif = one_hot_encode("ACGTACG")
	w = motif.shape[-1]

	X = _make_one_hot((100, 4, 101), random_state=12)
	center = (101 - w) // 2
	_plant(X, motif, [center] * 80)

	X_control = _make_one_hot((50, 4, 61), random_state=13)

	result = centrimo({'m1': motif.astype('float64')}, X,
		control_sequences=X_control, threshold=0.05)

	assert result.iloc[0]['n_control_sequences'] <= 50


def test_centrimo_control_does_not_bias_window_selection():
	# The enriched window is selected using the primary sequences alone, so
	# a control set with the motif planted strongly off-center (which would
	# pull the window elsewhere if it were allowed to influence selection)
	# must not change the window chosen for the primary set.
	n_seqs, seq_len = 200, 101
	motif = one_hot_encode("ACGTACG")
	w = motif.shape[-1]
	center = (seq_len - w) // 2

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=14)
	_plant(X, motif, [center] * 150)

	result_alone = centrimo({'m1': motif.astype('float64')}, X, threshold=0.05)

	X_control = _make_one_hot((n_seqs, 4, seq_len), random_state=15)
	_plant(X_control, motif, [5] * 150)

	result_with_control = centrimo({'m1': motif.astype('float64')}, X,
		control_sequences=X_control, threshold=0.05)

	assert (result_alone.iloc[0]['best_window_width'] ==
		result_with_control.iloc[0]['best_window_width'])
	assert (result_alone.iloc[0]['n_matching_sequences'] ==
		result_with_control.iloc[0]['n_matching_sequences'])


def test_centrimo_control_return_site_distances():
	motif = one_hot_encode("ACGTACG")
	X = _make_one_hot((10, 4, 50), random_state=16)
	X_control = _make_one_hot((8, 4, 50), random_state=17)

	result, distances, control_distances = centrimo(
		{'m1': motif.astype('float64')}, X, control_sequences=X_control,
		return_site_distances=True)

	assert distances.shape == (1, 10)
	assert control_distances.shape == (1, 8)


def test_centrimo_control_zero_sequences_raises():
	motif = one_hot_encode("ACGTACG")
	X = _make_one_hot((5, 4, 20), random_state=18)
	X_control = numpy.zeros((0, 4, 20), dtype='int8')

	assert_raises(ValueError, centrimo, {'m1': motif.astype('float64')}, X,
		X_control)


def test_centrimo_control_shorter_than_motif_raises():
	motif = one_hot_encode("ACGTACGTAC")  # width 10
	X = _make_one_hot((5, 4, 20), random_state=19)
	X_control = _make_one_hot((5, 4, 5), random_state=20)

	assert_raises(ValueError, centrimo, {'m1': motif.astype('float64')}, X,
		X_control)


##


def test_centrimo_separate_strands_reports_two_rows():
	motif = one_hot_encode("ACGTGCA")  # not self-reverse-complementary
	X = _make_one_hot((50, 4, 101), random_state=21)

	result = centrimo({'m1': motif.astype('float64')}, X,
		separate_strands=True)

	assert tuple(result.columns) == COLUMNS
	assert list(result['motif_name']) == ['m1', 'm1-rc']
	assert list(result['motif_idx']) == [0, 1]


def test_centrimo_separate_strands_detects_correct_strand():
	n_seqs, seq_len = 200, 101
	seq = "ACGTGCA"
	rc_seq = _reverse_complement_str(seq)

	motif = one_hot_encode(seq)
	rc_motif = one_hot_encode(rc_seq)
	w = motif.shape[-1]
	center = (seq_len - w) // 2

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=22)
	_plant(X, rc_motif, [center] * 150)

	result = centrimo({'m1': motif.astype('float64')}, X, threshold=0.05,
		separate_strands=True)

	fwd_row = result[result['motif_name'] == 'm1'].iloc[0]
	rc_row = result[result['motif_name'] == 'm1-rc'].iloc[0]

	assert rc_row['e_value'] < 1e-10
	assert rc_row['n_matching_sequences'] >= 0.9 * 150
	assert fwd_row['e_value'] > rc_row['e_value']


def test_centrimo_separate_strands_multiple_motifs():
	motifs = {
		'm1': one_hot_encode("ACGTGCA").astype('float64'),
		'm2': one_hot_encode("TTGCCAA").astype('float64'),
	}
	X = _make_one_hot((50, 4, 101), random_state=23)

	result = centrimo(motifs, X, separate_strands=True)

	assert list(result['motif_name']) == ['m1', 'm2', 'm1-rc', 'm2-rc']
	assert list(result['motif_idx']) == [0, 1, 2, 3]


def test_centrimo_separate_strands_noop_without_reverse_complement():
	motif = one_hot_encode("ACGTGCA")
	X = _make_one_hot((50, 4, 101), random_state=24)

	result = centrimo({'m1': motif.astype('float64')}, X,
		separate_strands=True, reverse_complement=False)

	assert tuple(result.columns) == COLUMNS
	assert list(result['motif_name']) == ['m1']


##


def test_centrimo_optimize_score_finds_stricter_threshold():
	# A loose nominal threshold lets in a lot of background noise alongside
	# the 100 planted exact matches; optimize_score should find that a much
	# stricter cutoff (closer to requiring the exact match) is far more
	# centrally enriched than the full, noisy nominal-threshold set.
	n_seqs, seq_len = 300, 101
	motif = one_hot_encode("ACGTGCA")
	w = motif.shape[-1]
	center = (seq_len - w) // 2

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=30)
	_plant(X, motif, [center] * 100)

	# A `threshold` this high is only "loose" once converted to CentriMo's
	# per-sequence-adjusted p-value (divided by `2 * n_valid_positions`,
	# `~190` here) -- it still comes out to a fairly strict effective
	# per-position requirement, but loose enough to admit a lot of noise
	# alongside the exact matches.
	default = centrimo({'m1': motif.astype('float64')}, X, threshold=1.0)
	optimized = centrimo({'m1': motif.astype('float64')}, X, threshold=1.0,
		optimize_score=True)

	assert tuple(optimized.columns) == COLUMNS + ('optimized_threshold_p_value',)

	row_default = default.iloc[0]
	row_opt = optimized.iloc[0]

	assert row_opt['optimized_threshold_p_value'] <= 0.05
	assert row_opt['p_value'] <= row_default['p_value']
	assert row_opt['n_sequences'] <= row_default['n_sequences']
	assert row_opt['e_value'] < row_default['e_value']


def test_centrimo_optimize_score_with_control_sequences():
	n_seqs, seq_len = 300, 101
	motif = one_hot_encode("ACGTGCA")
	w = motif.shape[-1]
	center = (seq_len - w) // 2

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=31)
	_plant(X, motif, [center] * 100)

	r = numpy.random.RandomState(32)
	X_control = _make_one_hot((n_seqs, 4, seq_len), random_state=32)
	offsets = r.choice([0, seq_len - w], size=100)
	_plant(X_control, motif, offsets)

	result = centrimo({'m1': motif.astype('float64')}, X,
		control_sequences=X_control, threshold=0.05, optimize_score=True)

	assert tuple(result.columns) == (COLUMNS + ('optimized_threshold_p_value',)
		+ CONTROL_COLUMNS[len(COLUMNS):])

	row = result.iloc[0]
	assert row['fisher_e_value'] < 1e-10


def test_centrimo_optimize_score_no_qualifying_sequences():
	# An extremely strict nominal threshold that nothing can pass makes
	# `score_thresholds` infinite, so `n_sequences` is deterministically 0
	# regardless of the random background -- this exercises the early-exit
	# branch's optimize_score column.
	motif = one_hot_encode("ACGTACGTAC")  # width 10
	X = _make_one_hot((10, 4, 20), random_state=33)

	result = centrimo({'m1': motif.astype('float64')}, X, threshold=1e-12,
		optimize_score=True)

	row = result.iloc[0]
	assert row['n_sequences'] == 0
	assert row['optimized_threshold_p_value'] == 1.0


def test_centrimo_optimize_score_max_score_thresholds_caps_grid():
	n_seqs, seq_len = 200, 101
	motif = one_hot_encode("ACGTGCA")
	w = motif.shape[-1]
	center = (seq_len - w) // 2

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=34)
	_plant(X, motif, [center] * 100)

	result = centrimo({'m1': motif.astype('float64')}, X, threshold=0.05,
		optimize_score=True, max_score_thresholds=2)

	row = result.iloc[0]
	assert not numpy.isnan(row['p_value'])
	assert 0 <= row['e_value'] <= 1


##


def test_centrimo_flip_requires_separate_strands():
	motif = one_hot_encode("ACGTGCA")
	X = _make_one_hot((10, 4, 50), random_state=35)

	assert_raises(ValueError, centrimo, {'m1': motif.astype('float64')}, X,
		None, flip=True)


def test_centrimo_flip_negates_only_rc_row_distances():
	# Plant the motif's reverse complement off-center, so a sign flip is
	# actually observable (planting at the exact center would give a
	# distance of 0, which is its own negation and wouldn't distinguish the
	# two cases).
	n_seqs, seq_len = 50, 101
	seq = "ACGTGCA"
	motif = one_hot_encode(seq)
	rc_motif = one_hot_encode(_reverse_complement_str(seq))
	offset = 10

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=36)
	_plant(X, rc_motif, [offset] * n_seqs)

	no_flip, dist_no_flip = centrimo({'m1': motif.astype('float64')}, X,
		threshold=0.05, separate_strands=True, return_site_distances=True)
	flipped, dist_flip = centrimo({'m1': motif.astype('float64')}, X,
		threshold=0.05, separate_strands=True, flip=True,
		return_site_distances=True)

	# The forward row is untouched by `flip`.
	assert_array_almost_equal(dist_no_flip[0], dist_flip[0])

	# The rc row is sign-negated wherever it has a real (non-NaN) value.
	valid = ~numpy.isnan(dist_no_flip[1])
	assert valid.sum() > 0
	assert_array_almost_equal(dist_flip[1, valid], -dist_no_flip[1, valid])

	# `flip` has no effect on any reported statistic.
	pandas.testing.assert_frame_equal(no_flip, flipped)


def test_centrimo_flip_no_effect_without_return_site_distances():
	motif = one_hot_encode("ACGTGCA")
	X = _make_one_hot((30, 4, 101), random_state=37)

	no_flip = centrimo({'m1': motif.astype('float64')}, X,
		separate_strands=True)
	flipped = centrimo({'m1': motif.astype('float64')}, X,
		separate_strands=True, flip=True)

	pandas.testing.assert_frame_equal(no_flip, flipped)
