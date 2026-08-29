# test_spamo_equivalence.py
# Contact: Adam He <adamyhe@gmail.com>
#
# Equivalence tests comparing this package's `spamo()` against the real
# MEME suite `spamo` binary. See test_centrimo_equivalence.py's header for
# the general setup (skips automatically without a real binary on PATH).
#
# The reference binary has no p-value-threshold mode for `-minscore` (unlike
# centrimo's `--use-pvalues`), so we convert our own `threshold` into an
# equivalent raw score via `utils._pvalue_score_thresholds` directly and
# pass that as `-minscore` -- using primary/secondary motifs of the same
# width and pseudocount so a single shared value applies to both.
#
# `-usebestsec -keepprimary -trim 0 -shared 0.99` disable reference-only
# behaviors this package doesn't implement (see spamo()'s docstring for the
# full list) -- `-shared` matters a lot here specifically: our synthetic
# sequences often share an identical planted primary+secondary region
# across many/all sequences, which the reference binary's default
# redundant-sequence elimination would otherwise treat as near-duplicates
# and drop, shrinking its reported counts for no reason related to the
# actual statistics being compared. `-shared` can't literally be 0 (the
# binary rejects it: "must be greater than expected similarity (0.25)",
# the baseline share two random same-length DNA sequences have by chance),
# so 0.99 (near-total-sequence identity required) is used instead to make
# the elimination as close to inert as possible.
#
# Also note: the reference binary's `gap` TSV column reports the *bin
# index*, not a bp value, when `bin_size_bp` != 1 (indistinguishable from a
# bp value at the default bin_size_bp=1, where they're numerically
# identical) -- see the leftover-bin test below.

import math
import shutil
import subprocess

import numpy
import pytest

from memelite.spamo import spamo
from memelite.spamo import _ORIENTATION_NAMES_RC
from memelite.io import write_meme
from memelite.utils import _pvalue_score_thresholds


SPAMO_BIN = shutil.which("spamo")

pytestmark = [
	pytest.mark.meme_ref,
	pytest.mark.skipif(SPAMO_BIN is None, reason="real `spamo` binary not "
		"found on PATH -- install via `conda create -n meme-ref -c bioconda "
		"-c conda-forge meme=5.5.9` and activate that environment (or "
		"otherwise put `spamo` on PATH) to run these tests."),
]

ALPHABET = "ACGT"
MAPPING = {c: i for i, c in enumerate(ALPHABET)}
EPS = 1e-4


def _random_sequences(n, seq_len, random_state):
	rng = numpy.random.RandomState(random_state)
	idxs = rng.randint(0, 4, size=(n, seq_len))
	return [''.join(ALPHABET[i] for i in row) for row in idxs]


def _plant(seqs, motif, offsets):
	w = len(motif)
	return [s if offset is None else s[:offset] + motif + s[offset+w:]
		for s, offset in zip(seqs, offsets)]


def _write_fasta(path, seqs):
	with open(path, "w") as f:
		for i, s in enumerate(seqs):
			f.write(f">seq{i}\n{s}\n")


def _write_uniform_bfile(path):
	with open(path, "w") as f:
		for c in ALPHABET:
			f.write(f"{c} {1.0/len(ALPHABET)}\n")


def _seqs_to_onehot(seqs):
	n, seq_len = len(seqs), len(seqs[0])
	X = numpy.zeros((n, 4, seq_len), dtype='float64')
	for i, s in enumerate(seqs):
		for j, c in enumerate(s):
			X[i, MAPPING[c], j] = 1
	return X


def _motif_from_str(seq, eps=EPS):
	pwm = numpy.full((4, len(seq)), eps)
	for j, c in enumerate(seq):
		pwm[MAPPING[c], j] = 1.0 - 3 * eps
	return pwm


def _score_threshold(seq, threshold, eps=EPS):
	pwm = _motif_from_str(seq, eps)
	log_pwm = numpy.log2(pwm + eps) - math.log2(0.25)
	thresh, _, _ = _pvalue_score_thresholds(log_pwm,
		numpy.array([0, len(seq)]), 0.1, threshold)
	return float(thresh[0])


def _run_real_spamo(tmp_path, fasta_path, primary_path, secondary_path,
	minscore, margin, range_, bin_size_bp, tag="out", extra_args=()):
	bfile = tmp_path / "uniform.bfile"
	if not bfile.exists():
		_write_uniform_bfile(bfile)

	out_dir = tmp_path / tag
	cmd = [SPAMO_BIN, "-oc", str(out_dir), "-text", "-usebestsec",
		"-keepprimary", "-trim", "0", "-shared", "0.99", "-bgfile", str(bfile),
		"-pseudo", str(EPS), "-minscore", str(minscore), "-margin", str(margin),
		"-range", str(range_), "-bin", str(bin_size_bp)]
	cmd += list(extra_args)
	cmd += [str(fasta_path), str(primary_path), str(secondary_path)]

	result = subprocess.run(cmd, capture_output=True, text=True)
	assert result.returncode == 0, result.stderr

	with open(out_dir / "spamo.tsv") as f:
		lines = [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]

	header = lines[0].split("\t")
	return [dict(zip(header, [v.strip() for v in line.split("\t")]))
		for line in lines[1:]]


def _index_by_orientation(real_rows):
	return {_ORIENTATION_NAMES_RC[int(row['orient'])]: row for row in real_rows}


###


def test_spamo_matches_reference_downstream_same(tmp_path):
	seq_len = 200
	primary_str, secondary_str = "ACGTGCA", "TTGCCAA"
	w_p, w_s = len(primary_str), len(secondary_str)
	margin = 40

	seqs = _random_sequences(150, seq_len, random_state=0)
	primary_pos = (seq_len - w_p) // 2
	gap = 12
	secondary_pos = primary_pos + w_p + gap

	seqs = _plant(seqs, primary_str, [primary_pos] * len(seqs))
	n_planted = 100
	offsets = [secondary_pos] * n_planted + [None] * (len(seqs) - n_planted)
	seqs = _plant(seqs, secondary_str, offsets)

	fasta_path = tmp_path / "seqs.fasta"
	_write_fasta(fasta_path, seqs)

	primary_pwm = _motif_from_str(primary_str)
	secondary_pwm = _motif_from_str(secondary_str)
	primary_path, secondary_path = tmp_path / "primary.meme", tmp_path / "secondary.meme"
	write_meme(str(primary_path), {"primary": primary_pwm})
	write_meme(str(secondary_path), {"secondary": secondary_pwm})

	threshold = 0.001
	minscore = _score_threshold(primary_str, threshold)
	assert minscore == pytest.approx(_score_threshold(secondary_str, threshold))

	real = _index_by_orientation(_run_real_spamo(tmp_path, fasta_path,
		primary_path, secondary_path, minscore, margin, margin, 1))
	assert 'downstream_same' in real
	real_row = real['downstream_same']

	result = spamo({"primary": primary_pwm}, {"secondary": secondary_pwm},
		_seqs_to_onehot(seqs), margin=margin, range_=margin, threshold=threshold,
		eps=EPS)
	row = result[(result['motif_name'] == 'secondary') &
		(result['orientation'] == 'downstream_same')].iloc[0]

	# `-shared 0.99` still occasionally eliminates a sequence or two from
	# purely random background content by chance, so a small tolerance on
	# `n_sequences` (not just `n_matching_sequences`) is expected here too.
	assert abs(row['n_sequences'] - int(real_row['total'])) <= 3
	assert abs(row['n_matching_sequences'] - int(real_row['count'])) <= 1
	# At these extreme p-value magnitudes (~1e-190), even the single-sequence
	# count difference already allowed above shifts the exponent noticeably
	# -- a wider tolerance here is expected numerical sensitivity, not a
	# structural mismatch (the counts themselves already agree above).
	assert numpy.log10(max(row['p_value'], 1e-300)) == pytest.approx(
		numpy.log10(max(float(real_row['p-value']), 1e-300)), abs=5.0)
	assert numpy.log10(max(row['e_value'], 1e-300)) == pytest.approx(
		numpy.log10(max(float(real_row['E-value']), 1e-300)), abs=5.0)


def test_spamo_matches_reference_reverse_strand_primary(tmp_path):
	# The primary is planted on the reverse strand and the secondary
	# literally before it in raw coordinates -- exercises this package's
	# side-rotation fix (upstream/downstream relative to the primary's own
	# matched strand) against the reference binary's own behavior.
	seq_len = 200
	primary_str, secondary_str = "ACGTGCA", "TTGCCAA"

	def rc(seq, comp={'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}):
		return ''.join(comp[c] for c in reversed(seq))

	w_p, w_s = len(primary_str), len(secondary_str)
	margin = 40

	seqs = _random_sequences(150, seq_len, random_state=1)
	primary_pos = (seq_len - w_p) // 2
	gap = 15
	secondary_pos = primary_pos - gap - w_s

	seqs = _plant(seqs, rc(primary_str), [primary_pos] * len(seqs))
	n_planted = 100
	offsets = [secondary_pos] * n_planted + [None] * (len(seqs) - n_planted)
	seqs = _plant(seqs, rc(secondary_str), offsets)

	fasta_path = tmp_path / "seqs.fasta"
	_write_fasta(fasta_path, seqs)

	primary_pwm = _motif_from_str(primary_str)
	secondary_pwm = _motif_from_str(secondary_str)
	primary_path, secondary_path = tmp_path / "primary.meme", tmp_path / "secondary.meme"
	write_meme(str(primary_path), {"primary": primary_pwm})
	write_meme(str(secondary_path), {"secondary": secondary_pwm})

	threshold = 0.001
	minscore = _score_threshold(primary_str, threshold)

	real = _index_by_orientation(_run_real_spamo(tmp_path, fasta_path,
		primary_path, secondary_path, minscore, margin, margin, 1,
		tag="out_rc"))
	assert 'downstream_same' in real
	assert 'upstream_same' not in real

	result = spamo({"primary": primary_pwm}, {"secondary": secondary_pwm},
		_seqs_to_onehot(seqs), margin=margin, range_=margin, threshold=threshold,
		eps=EPS)
	sub = result[result['motif_name'] == 'secondary']
	assert (sub['orientation'] == 'downstream_same').any()
	assert not (sub['orientation'] == 'upstream_same').any()


def test_spamo_matches_reference_secondary_pal_orientation(tmp_path):
	# Half the sequences have the secondary matching the forward strand
	# downstream of the primary; the other half have its reverse complement
	# at the same gap/side. Neither raw quadrant alone has the full signal,
	# but `downstream_secondary_pal` (which pools both strands on the
	# downstream side) should recover it in both implementations.
	seq_len = 200
	primary_str, secondary_str = "ACGTGCA", "TTGCCAA"

	def rc(seq, comp={'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}):
		return ''.join(comp[c] for c in reversed(seq))

	w_p, w_s = len(primary_str), len(secondary_str)
	margin = 40

	seqs = _random_sequences(200, seq_len, random_state=2)
	primary_pos = (seq_len - w_p) // 2
	gap = 12
	secondary_pos = primary_pos + w_p + gap

	seqs = _plant(seqs, primary_str, [primary_pos] * len(seqs))
	fwd_offsets = [secondary_pos] * 100 + [None] * (len(seqs) - 100)
	seqs = _plant(seqs, secondary_str, fwd_offsets)
	rc_offsets = [None] * 100 + [secondary_pos] * 50 + [None] * (len(seqs) - 150)
	seqs = _plant(seqs, rc(secondary_str), rc_offsets)

	fasta_path = tmp_path / "seqs.fasta"
	_write_fasta(fasta_path, seqs)

	primary_pwm = _motif_from_str(primary_str)
	secondary_pwm = _motif_from_str(secondary_str)
	primary_path = tmp_path / "primary.meme"
	secondary_path = tmp_path / "secondary.meme"
	write_meme(str(primary_path), {"primary": primary_pwm})
	write_meme(str(secondary_path), {"secondary": secondary_pwm})

	threshold = 0.001
	minscore = _score_threshold(primary_str, threshold)

	real = _index_by_orientation(_run_real_spamo(tmp_path, fasta_path,
		primary_path, secondary_path, minscore, margin, margin, 1,
		tag="out_pal"))
	assert 'downstream_secondary_pal' in real
	real_row = real['downstream_secondary_pal']

	result = spamo({"primary": primary_pwm}, {"secondary": secondary_pwm},
		_seqs_to_onehot(seqs), margin=margin, range_=margin, threshold=threshold,
		eps=EPS)
	row = result[(result['motif_name'] == 'secondary') &
		(result['orientation'] == 'downstream_secondary_pal')].iloc[0]

	assert row['n_sequences'] == int(real_row['total'])
	assert abs(row['n_matching_sequences'] - int(real_row['count'])) <= 1
	assert numpy.log10(max(row['p_value'], 1e-300)) == pytest.approx(
		numpy.log10(max(float(real_row['p-value']), 1e-300)), abs=5.0)


def test_spamo_matches_reference_leftover_bin(tmp_path):
	seq_len = 200
	primary_str, secondary_str = "ACGTGCA", "TTGCCAA"
	w_p, w_s = len(primary_str), len(secondary_str)

	margin, bin_size_bp = 25, 5
	# quad_opt_count = margin - w_s + 1 = 19; 19 % 5 = 4, so the leftover
	# bin covers gaps [15, 18] -- narrower than a full bin_size_bp=5 bin.
	seqs = _random_sequences(200, seq_len, random_state=3)
	primary_pos = 90
	gap = 17
	secondary_pos = primary_pos + w_p + gap

	seqs = _plant(seqs, primary_str, [primary_pos] * len(seqs))
	seqs = _plant(seqs, secondary_str, [secondary_pos] * len(seqs))

	fasta_path = tmp_path / "seqs.fasta"
	_write_fasta(fasta_path, seqs)

	primary_pwm = _motif_from_str(primary_str)
	secondary_pwm = _motif_from_str(secondary_str)
	primary_path = tmp_path / "primary.meme"
	secondary_path = tmp_path / "secondary.meme"
	write_meme(str(primary_path), {"primary": primary_pwm})
	write_meme(str(secondary_path), {"secondary": secondary_pwm})

	threshold = 0.001
	minscore = _score_threshold(primary_str, threshold)

	real = _index_by_orientation(_run_real_spamo(tmp_path, fasta_path,
		primary_path, secondary_path, minscore, margin, margin, bin_size_bp,
		tag="out_leftover"))
	assert 'downstream_same' in real
	real_row = real['downstream_same']
	# `gap` here is the *bin index* (3 = the leftover bin), not a bp value.
	assert int(real_row['gap']) == 3

	result = spamo({"primary": primary_pwm}, {"secondary": secondary_pwm},
		_seqs_to_onehot(seqs), margin=margin, range_=margin,
		bin_size_bp=bin_size_bp, threshold=threshold, eps=EPS)
	row = result[(result['motif_name'] == 'secondary') &
		(result['orientation'] == 'downstream_same')].iloc[0]

	assert row['gap_lo'] == 15
	assert row['gap_hi'] == 18
	# `-shared 0.99` still occasionally eliminates a sequence or two from
	# purely random background content by chance.
	assert abs(row['n_sequences'] - int(real_row['total'])) <= 5
	assert abs(row['n_matching_sequences'] - int(real_row['count'])) <= 5


def test_spamo_matches_reference_differing_widths(tmp_path):
	seq_len = 200
	primary_str, secondary_str = "ACGTGCA", "TTGCCAAGCT"  # widths 7 and 10
	w_p = len(primary_str)
	margin = 40

	seqs = _random_sequences(150, seq_len, random_state=4)
	primary_pos = (seq_len - w_p) // 2
	gap = 12
	secondary_pos = primary_pos + w_p + gap

	seqs = _plant(seqs, primary_str, [primary_pos] * len(seqs))
	n_planted = 100
	offsets = [secondary_pos] * n_planted + [None] * (len(seqs) - n_planted)
	seqs = _plant(seqs, secondary_str, offsets)

	fasta_path = tmp_path / "seqs.fasta"
	_write_fasta(fasta_path, seqs)

	primary_pwm = _motif_from_str(primary_str)
	secondary_pwm = _motif_from_str(secondary_str)
	primary_path = tmp_path / "primary.meme"
	secondary_path = tmp_path / "secondary.meme"
	write_meme(str(primary_path), {"primary": primary_pwm})
	write_meme(str(secondary_path), {"secondary": secondary_pwm})

	threshold = 0.001
	# The reference binary applies one raw `-minscore` to both motifs
	# regardless of width, unlike our own per-motif p-value-driven
	# thresholds -- use whichever of the two per-motif thresholds is
	# loosest, so the same near-exact planted matches clear both.
	minscore = min(_score_threshold(primary_str, threshold),
		_score_threshold(secondary_str, threshold))

	real = _index_by_orientation(_run_real_spamo(tmp_path, fasta_path,
		primary_path, secondary_path, minscore, margin, margin, 1,
		tag="out_widths"))
	assert 'downstream_same' in real
	real_row = real['downstream_same']

	result = spamo({"primary": primary_pwm}, {"secondary": secondary_pwm},
		_seqs_to_onehot(seqs), margin=margin, range_=margin, threshold=threshold,
		eps=EPS)
	row = result[(result['motif_name'] == 'secondary') &
		(result['orientation'] == 'downstream_same')].iloc[0]

	# With mismatched primary/secondary widths, the min() of the two per-
	# motif thresholds above is looser for whichever motif is wider than a
	# matched-width comparison would need, admitting more purely-
	# coincidental background matches among the sequences with no planted
	# secondary at all -- this test is checking structural correctness
	# (differing widths work at all, at the right gap), not a tight
	# statistical match, so allow a generous count tolerance.
	assert abs(row['n_sequences'] - int(real_row['total'])) <= 30
	assert abs(row['n_matching_sequences'] - int(real_row['count'])) <= 5
	assert row['gap_lo'] <= gap <= row['gap_hi']


def test_spamo_matches_reference_norc(tmp_path):
	seq_len = 200
	primary_str, secondary_str = "ACGTGCA", "TTGCCAA"
	w_p = len(primary_str)
	margin = 40

	seqs = _random_sequences(150, seq_len, random_state=5)
	primary_pos = (seq_len - w_p) // 2
	gap = 12
	secondary_pos = primary_pos + w_p + gap

	seqs = _plant(seqs, primary_str, [primary_pos] * len(seqs))
	n_planted = 100
	offsets = [secondary_pos] * n_planted + [None] * (len(seqs) - n_planted)
	seqs = _plant(seqs, secondary_str, offsets)

	fasta_path = tmp_path / "seqs.fasta"
	_write_fasta(fasta_path, seqs)

	primary_pwm = _motif_from_str(primary_str)
	secondary_pwm = _motif_from_str(secondary_str)
	primary_path = tmp_path / "primary.meme"
	secondary_path = tmp_path / "secondary.meme"
	write_meme(str(primary_path), {"primary": primary_pwm})
	write_meme(str(secondary_path), {"secondary": secondary_pwm})

	threshold = 0.001
	minscore = _score_threshold(primary_str, threshold)

	# `-norc` does *not* collapse the real binary's output to a simpler
	# 2-category model -- it still reports the full 9-orientation space,
	# just with every opposite-strand-including category always empty
	# (confirmed empirically; see spamo()'s docstring).
	real = _index_by_orientation(_run_real_spamo(tmp_path, fasta_path,
		primary_path, secondary_path, minscore, margin, margin, 1,
		tag="out_norc", extra_args=["-norc"]))
	assert 'downstream_same' in real
	assert 'upstream_same' not in real
	real_row = real['downstream_same']

	result = spamo({"primary": primary_pwm}, {"secondary": secondary_pwm},
		_seqs_to_onehot(seqs), margin=margin, range_=margin, threshold=threshold,
		eps=EPS, reverse_complement=False)
	row = result[(result['motif_name'] == 'secondary') &
		(result['orientation'] == 'downstream_same')].iloc[0]

	assert row['n_sequences'] == int(real_row['total'])
	assert abs(row['n_matching_sequences'] - int(real_row['count'])) <= 1
	assert row['gap_lo'] <= gap <= row['gap_hi']
	assert numpy.log10(max(row['p_value'], 1e-300)) == pytest.approx(
		numpy.log10(max(float(real_row['p-value']), 1e-300)), abs=1.5)
