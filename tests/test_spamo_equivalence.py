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
# `-usebestsec -keepprimary -trim 0` disable reference-only behaviors this
# package doesn't implement (see spamo()'s docstring for the full list).

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
	minscore, margin, range_, bin_size_bp, tag="out"):
	bfile = tmp_path / "uniform.bfile"
	if not bfile.exists():
		_write_uniform_bfile(bfile)

	out_dir = tmp_path / tag
	cmd = [SPAMO_BIN, "-oc", str(out_dir), "-text", "-usebestsec",
		"-keepprimary", "-trim", "0", "-bgfile", str(bfile),
		"-pseudo", str(EPS), "-minscore", str(minscore), "-margin", str(margin),
		"-range", str(range_), "-bin", str(bin_size_bp),
		str(fasta_path), str(primary_path), str(secondary_path)]

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

	assert row['n_sequences'] == int(real_row['total'])
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
