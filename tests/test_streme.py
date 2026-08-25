# test_streme.py
# Contact: Adam He <adamyhe@gmail.com>

import numpy
import pandas
import pytest

from memelite.streme import streme
from memelite.streme import _streme_best_sites
from memelite.streme import _streme_erase
from memelite.streme import _code_to_pwm
from memelite.utils import one_hot_encode
from memelite.utils import characters
from memelite.utils import _kmer_codes
from memelite.utils import _reverse_complement_kmer_codes
from memelite.utils import _markov_transition_probs
from memelite.utils import _markov_shuffle_sequences

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


COLUMNS = ('motif_name', 'motif_idx', 'width', 'consensus',
	'n_primary_sites', 'n_control_sites', 'score_threshold', 'p_value',
	'e_value')


def _make_one_hot(shape, random_state=None):
	"""Build a correctly-formed random one-hot array with a fixed random
	state, all sequences the same length."""

	random_state = numpy.random.RandomState(random_state)

	n, c, l = shape
	idxs = random_state.randint(0, c, size=(n, l))

	ohe = numpy.zeros(shape, dtype='int8')
	for i in range(n):
		ohe[i, idxs[i], numpy.arange(l)] = 1

	return ohe


def _reverse_complement_str(seq, complement={'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}):
	return ''.join(complement[c] for c in reversed(seq))


def _plant(X, motif, offsets, mask=None):
	"""Overwrite each selected sequence in `X` with `motif` at its offset."""

	w = motif.shape[-1]
	idxs = range(len(offsets)) if mask is None else numpy.where(mask)[0]
	for i in idxs:
		X[i, :, offsets[i]:offsets[i]+w] = motif
	return X


###


def test_streme_recovers_planted_motif():
	random_state = numpy.random.RandomState(0)

	n_seqs, seq_len = 300, 100
	motif = one_hot_encode("ACGTACGTA")
	w = motif.shape[-1]

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=0)
	mask = random_state.random_sample(n_seqs) < 0.6
	offsets = random_state.randint(5, seq_len - w - 5, size=n_seqs)
	_plant(X, motif, offsets, mask)

	motifs, results = streme(X, minw=6, maxw=12, random_state=0, patience=3)

	assert tuple(results.columns) == COLUMNS
	assert len(motifs) >= 1
	assert results.loc[0, 'p_value'] <= 0.05

	consensus = characters(motifs['STREME-1'], force=True)
	rc = _reverse_complement_str(consensus)
	assert ("ACGTACGTA" in consensus) or ("ACGTACGTA" in rc) or \
		(consensus in "ACGTACGTA") or (rc in "ACGTACGTA")


def _shares_long_common_substring(a, b, min_len):
	"""Whether `a` and `b` share a common substring of at least `min_len`."""

	for i in range(len(a) - min_len + 1):
		if a[i:i+min_len] in b:
			return True
	return False


def test_streme_recovers_reverse_complement_motif():
	random_state = numpy.random.RandomState(1)

	n_seqs, seq_len = 300, 100
	fwd = "GATTACAGA"
	rc_str = _reverse_complement_str(fwd)
	motif = one_hot_encode(rc_str)
	w = motif.shape[-1]

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=1)
	mask = random_state.random_sample(n_seqs) < 0.6
	offsets = random_state.randint(5, seq_len - w - 5, size=n_seqs)
	_plant(X, motif, offsets, mask)

	motifs, results = streme(X, minw=6, maxw=12, random_state=1, patience=3)

	assert len(motifs) >= 1
	consensus = characters(motifs['STREME-1'], force=True)
	crc = _reverse_complement_str(consensus)

	# The discovered width need not exactly match the planted width (a
	# found candidate can be shifted by a base or two while still covering
	# most of the real site), so check for a long shared substring with
	# whichever literal string was actually planted (`rc_str`) rather than
	# requiring an exact match against either orientation.
	assert _shares_long_common_substring(rc_str, consensus, w - 2) or \
		_shares_long_common_substring(rc_str, crc, w - 2)


def test_streme_no_signal_finds_nothing():
	X = _make_one_hot((300, 4, 100), random_state=2)
	motifs, results = streme(X, minw=6, maxw=10, random_state=2, patience=3)

	assert len(motifs) == 0
	assert len(results) == 0
	assert tuple(results.columns) == COLUMNS


def test_streme_two_distinct_motifs_recovered():
	random_state = numpy.random.RandomState(3)

	n_seqs, seq_len = 400, 120
	motif1 = one_hot_encode("GATTACAGG")
	motif2 = one_hot_encode("TTCCGAACA")
	w = motif1.shape[-1]

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=3)
	mask1 = random_state.random_sample(n_seqs) < 0.6
	mask2 = random_state.random_sample(n_seqs) < 0.6
	offsets1 = random_state.randint(5, seq_len - w - 5, size=n_seqs)
	offsets2 = random_state.randint(5, seq_len - w - 5, size=n_seqs)
	_plant(X, motif1, offsets1, mask1)
	_plant(X, motif2, offsets2, mask2)

	motifs, results = streme(X, minw=6, maxw=10, random_state=3, patience=3)

	assert len(motifs) >= 2
	assert list(results['motif_name'][:2]) == ['STREME-1', 'STREME-2']
	# e_value = p_value * rank (per the real STREME binary's formula) for
	# every row.
	for i, row in results.iterrows():
		assert row['e_value'] == pytest.approx(min(row['p_value'] * (i + 1), 1.0))


def test_streme_nmotifs_forces_fixed_count_regardless_of_significance():
	X = _make_one_hot((150, 4, 80), random_state=4)
	motifs, results = streme(X, minw=6, maxw=8, random_state=4, nmotifs=2)

	assert len(motifs) == 2
	assert len(results) == 2
	assert list(motifs.keys()) == ['STREME-1', 'STREME-2']


def test_streme_explicit_control_sequences():
	random_state = numpy.random.RandomState(5)

	n_seqs, seq_len = 300, 100
	motif = one_hot_encode("ACGTTGCAA")
	w = motif.shape[-1]

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=5)
	mask = random_state.random_sample(n_seqs) < 0.7
	offsets = random_state.randint(5, seq_len - w - 5, size=n_seqs)
	_plant(X, motif, offsets, mask)

	control = _make_one_hot((n_seqs, 4, seq_len), random_state=55)

	motifs, results = streme(X, control_sequences=control, minw=6, maxw=12,
		random_state=5, patience=3)

	assert len(motifs) >= 1
	assert results.loc[0, 'p_value'] <= 0.05


def test_streme_variable_length_string_input():
	random_state = numpy.random.RandomState(6)
	alphabet = numpy.array(['A', 'C', 'G', 'T'])

	motif = "ACGTACGTA"
	seqs = []
	for i in range(200):
		length = random_state.randint(80, 150)
		idxs = random_state.randint(0, 4, size=length)
		seq = ''.join(alphabet[idxs])
		if random_state.random_sample() < 0.6:
			pos = random_state.randint(5, length - len(motif) - 5)
			seq = seq[:pos] + motif + seq[pos+len(motif):]
		seqs.append(seq)

	motifs, results = streme(seqs, minw=6, maxw=12, random_state=6, patience=3)

	assert len(motifs) >= 1


def test_streme_return_site_positions():
	random_state = numpy.random.RandomState(7)

	n_seqs, seq_len = 250, 90
	motif = one_hot_encode("ACGTACGTA")
	w = motif.shape[-1]

	X = _make_one_hot((n_seqs, 4, seq_len), random_state=7)
	mask = random_state.random_sample(n_seqs) < 0.7
	offsets = random_state.randint(5, seq_len - w - 5, size=n_seqs)
	_plant(X, motif, offsets, mask)

	motifs, results, site_positions = streme(X, minw=6, maxw=12,
		random_state=7, patience=3, return_site_positions=True)

	assert len(site_positions) == len(motifs)
	for entry in site_positions:
		assert entry['motif_name'] in motifs
		assert 'search_primary_positions' in entry
		assert 'search_primary_strands' in entry
		assert 'holdout_primary_positions' in entry
		assert 'holdout_primary_strands' in entry


def test_streme_minw_greater_than_maxw_raises():
	X = _make_one_hot((50, 4, 60), random_state=8)
	assert_raises(ValueError, streme, X, minw=12, maxw=8)


def test_streme_no_sequence_long_enough_raises():
	X = _make_one_hot((50, 4, 5), random_state=9)
	assert_raises(ValueError, streme, X, minw=8, maxw=15)


def test_streme_zero_sequences_raises():
	assert_raises(ValueError, streme, [])


def test_streme_zero_control_sequences_raises():
	X = _make_one_hot((50, 4, 60), random_state=10)
	assert_raises(ValueError, streme, X, control_sequences=[])


def test_streme_too_few_sequences_for_holdout_raises():
	X = _make_one_hot((1, 4, 60), random_state=11)
	assert_raises(ValueError, streme, X, hofract=0.5)


###
# Unit tests on the smaller building blocks.
###


def test_kmer_codes_basic():
	# 'ACGT' with alphabet A=0,C=1,G=2,T=3 and w=2 should give codes
	# AC=0*4+1=1, CG=1*4+2=6, GT=2*4+3=11.
	X = numpy.array([0, 1, 2, 3], dtype=numpy.int8)
	X_lengths = numpy.array([0, 4], dtype=numpy.int64)

	codes, owners = _kmer_codes(X, X_lengths, 2, 4)
	assert_array_equal(codes, [1, 6, 11])
	assert_array_equal(owners, [0, 0, 0])


def test_kmer_codes_skips_unknown_and_erased():
	X = numpy.array([0, 1, -1, 3, -2], dtype=numpy.int8)
	X_lengths = numpy.array([0, 5], dtype=numpy.int64)

	codes, owners = _kmer_codes(X, X_lengths, 2, 4)
	# windows: [0,1]->valid, [1,-1]->invalid, [-1,3]->invalid, [3,-2]->invalid
	assert_array_equal(codes, [1, -1, -1, -1])


def test_reverse_complement_kmer_codes_self_consistent():
	# ACGT (0,1,2,3) reverse-complemented is ACGT again (A<->T, C<->G).
	X = numpy.array([0, 1, 2, 3], dtype=numpy.int8)
	X_lengths = numpy.array([0, 4], dtype=numpy.int64)
	codes, _ = _kmer_codes(X, X_lengths, 4, 4)

	rc = _reverse_complement_kmer_codes(codes, 4, 4)
	assert_array_equal(rc, codes)

	# Applying it twice returns the original code.
	rc2 = _reverse_complement_kmer_codes(rc, 4, 4)
	assert_array_equal(rc2, codes)


def test_code_to_pwm_roundtrips_through_characters():
	code = 0
	for base in [0, 1, 2, 3, 0, 3]:
		code = code * 4 + base

	pwm = _code_to_pwm(code, 6, 4, eps=0.0001)
	assert characters(pwm, force=True) == "ACGTAT"


def test_markov_shuffle_preserves_length_and_order0_composition():
	random_state = numpy.random.RandomState(12)
	X = random_state.randint(0, 4, size=2000).astype(numpy.int8)
	X_lengths = numpy.array([0, 1000, 2000], dtype=numpy.int64)

	marginal, transitions = _markov_transition_probs(X, X_lengths, 0, 4, 1e-3)
	shuffled = _markov_shuffle_sequences(X_lengths, 0, 4, marginal, transitions, 42)

	assert shuffled.shape == X.shape
	original_freq = numpy.bincount(X, minlength=4) / len(X)
	shuffled_freq = numpy.bincount(shuffled, minlength=4) / len(shuffled)
	assert_array_almost_equal(original_freq, shuffled_freq, decimal=1)


def test_streme_erase_masks_only_positive_positions():
	alphabet_size = 4
	w = 4

	# A PWM where only the first two positions have a positive log-odds
	# contribution for base 0, and the last two are negative for base 0.
	pwm_fwd = numpy.array([
		[1.0, 1.0, -1.0, -1.0],
		[-1.0, -1.0, -1.0, -1.0],
		[-1.0, -1.0, -1.0, -1.0],
		[-1.0, -1.0, -1.0, -1.0],
	])
	pwm_rc = pwm_fwd[::-1, ::-1].copy()

	X = numpy.array([0, 0, 0, 0], dtype=numpy.int8)
	X_lengths = numpy.array([0, 4], dtype=numpy.int64)
	scores = numpy.array([10.0])
	positions = numpy.array([0], dtype=numpy.int32)
	strands = numpy.array([0], dtype=numpy.int8)

	_streme_erase(X, X_lengths, scores, positions, strands, pwm_fwd, pwm_rc,
		w, threshold=0.0)

	assert_array_equal(X, [-2, -2, 0, 0])


def test_streme_erase_falls_back_to_middle_position():
	alphabet_size = 4
	w = 4

	# No position has a positive contribution for base 0 anywhere.
	pwm_fwd = numpy.full((alphabet_size, w), -1.0)
	pwm_rc = pwm_fwd[::-1, ::-1].copy()

	X = numpy.array([0, 0, 0, 0], dtype=numpy.int8)
	X_lengths = numpy.array([0, 4], dtype=numpy.int64)
	scores = numpy.array([10.0])
	positions = numpy.array([0], dtype=numpy.int32)
	strands = numpy.array([0], dtype=numpy.int8)

	_streme_erase(X, X_lengths, scores, positions, strands, pwm_fwd, pwm_rc,
		w, threshold=0.0)

	assert_array_equal(X, [0, 0, -2, 0])


def test_streme_best_sites_disqualifies_erased_windows():
	w = 3
	pwm_fwd = numpy.zeros((4, w))
	pwm_rc = pwm_fwd[::-1, ::-1].copy()

	# Sequence: [0,1,2, -2, 0,1,2] -- the erased base at index 3 should
	# disqualify every window overlapping it (positions 1, 2, and 3), but
	# not the fully-clean windows at positions 0 and 4.
	X = numpy.array([0, 1, 2, -2, 0, 1, 2], dtype=numpy.int8)
	X_lengths = numpy.array([0, 7], dtype=numpy.int64)

	scores, positions, strands = _streme_best_sites(X, X_lengths, pwm_fwd,
		pwm_rc, w, reverse_complement=False)

	assert positions[0] in (0, 4)
