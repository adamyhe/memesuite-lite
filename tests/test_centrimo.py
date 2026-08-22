# test_centrimo.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

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

	result = centrimo({'m1': motif.astype('float64')}, X, threshold=0.001)

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

	result = centrimo({'m1': motif.astype('float64')}, X, threshold=0.001)

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

	with_rc = centrimo({'m1': motif.astype('float64')}, X, threshold=0.001,
		reverse_complement=True)
	without_rc = centrimo({'m1': motif.astype('float64')}, X, threshold=0.001,
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

	distances = _centrimo_best_sites(X, 1, 10, pwm, pwm_lengths,
		score_thresholds, False, 1)

	expected_center = (2 + 3) / 2
	expected_distance = expected_center - (10 - 1) / 2.0
	assert_array_almost_equal(distances, [[expected_distance]])


def test_centrimo_below_threshold_excluded():
	seq_len = 20
	motif = one_hot_encode("ACGTACG")
	w = motif.shape[-1]

	X = _make_one_hot((2, 4, seq_len), random_state=3)
	_plant(X, motif, [5])

	result, distances = centrimo({'m1': motif.astype('float64')}, X,
		threshold=0.001, return_site_distances=True)

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

	result = centrimo(motifs, X, threshold=0.001)

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
