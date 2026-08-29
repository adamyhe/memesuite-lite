# test_centrimo_equivalence.py
# Contact: Adam He <adamyhe@gmail.com>
#
# Equivalence tests comparing this package's `centrimo()` against the real
# MEME suite `centrimo` binary. These are NOT run by default -- they require
# a real `centrimo` binary on PATH (e.g. `conda create -n meme-ref -c
# bioconda -c conda-forge meme=5.5.9 && conda activate meme-ref`) and are
# skipped automatically if one isn't found.
#
# Known, deliberate divergences from the reference binary NOT covered here
# (see `centrimo()`'s docstring for the full list):
#   - pseudocount convention: the reference binary's `--motif-pseudo` does an
#     nsites-weighted Bayesian blend; we always just add a flat `eps` before
#     taking the log. We sidestep this by passing `--motif-pseudo` equal to
#     our own `eps` (the two converge as both go to 0) and by only testing
#     strict/typical thresholds, where this makes a negligible difference.
#   - bin_size discretization: our score<->p-value mapping is binned
#     (`bin_size`, default 0.1), so results right at a significance boundary
#     can differ by one sequence from the reference binary's own (different)
#     internal discretization -- tolerances below allow for this.

import shutil
import subprocess

import numpy
import pandas
import pytest

from memelite.centrimo import centrimo
from memelite.io import write_meme


CENTRIMO_BIN = shutil.which("centrimo")

pytestmark = [
	pytest.mark.meme_ref,
	pytest.mark.skipif(CENTRIMO_BIN is None, reason="real `centrimo` binary "
		"not found on PATH -- install via `conda create -n meme-ref -c "
		"bioconda -c conda-forge meme=5.5.9` and activate that environment "
		"(or otherwise put `centrimo` on PATH) to run these tests."),
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


def _run_real_centrimo(tmp_path, fasta_path, meme_path, score, neg_fasta=None,
	extra_args=(), tag="out"):
	bfile = tmp_path / "uniform.bfile"
	if not bfile.exists():
		_write_uniform_bfile(bfile)

	out_dir = tmp_path / tag
	cmd = [CENTRIMO_BIN, "--oc", str(out_dir), "--use-pvalues",
		"--score", str(score), "--bfile", str(bfile),
		"--motif-pseudo", str(EPS)]
	if neg_fasta is not None:
		cmd += ["--neg", str(neg_fasta)]
	cmd += list(extra_args)
	cmd += [str(fasta_path), str(meme_path)]

	result = subprocess.run(cmd, capture_output=True, text=True)
	assert result.returncode == 0, result.stderr

	with open(out_dir / "centrimo.tsv") as f:
		lines = [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]

	header = lines[0].split("\t")
	return [dict(zip(header, [v.strip() for v in line.split("\t")]))
		for line in lines[1:]]


###


@pytest.mark.parametrize("seq_len,motif_str", [
	(41, "ACGTACGT"),   # n_valid_positions = 34 (even)
	(42, "ACGTACGT"),   # n_valid_positions = 35 (odd)
])
def test_centrimo_matches_reference_basic(tmp_path, seq_len, motif_str):
	w = len(motif_str)
	seqs = _random_sequences(80, seq_len, random_state=0)
	n_planted = 40
	center = (seq_len - w) // 2
	offsets = [center] * n_planted + [None] * (len(seqs) - n_planted)
	seqs = _plant(seqs, motif_str, offsets)

	fasta_path = tmp_path / "seqs.fasta"
	_write_fasta(fasta_path, seqs)

	pwm = _motif_from_str(motif_str)
	meme_path = tmp_path / "motif.meme"
	write_meme(str(meme_path), {"m1": pwm})

	threshold = 0.05
	real = _run_real_centrimo(tmp_path, fasta_path, meme_path, threshold)
	assert len(real) == 1
	real = real[0]

	result = centrimo({"m1": pwm}, _seqs_to_onehot(seqs), threshold=threshold,
		eps=EPS)
	row = result.iloc[0]

	assert row["n_sequences"] == int(real["total_sites"])
	assert abs(row["n_matching_sequences"] - int(real["sites_in_bin"])) <= 1
	assert row["best_window_width"] == float(real["bin_width"])
	# At the tightest possible bin (bin_width=1, radius=0), a single
	# sequence's best site can fall on either side of an exact tie-break
	# boundary, which shifts the p-value/e-value noticeably more than at
	# looser bins -- allow a wider tolerance to accommodate that.
	assert numpy.log10(row["p_value"]) == pytest.approx(
		numpy.log10(float(real["p-value"])), abs=1.5)
	assert numpy.log10(row["e_value"]) == pytest.approx(
		numpy.log10(float(real["E-value"])), abs=1.5)


def test_centrimo_matches_reference_neg(tmp_path):
	seq_len, w = 41, 8
	motif_str = "ACGTACGT"
	center = (seq_len - w) // 2

	pos_seqs = _random_sequences(100, seq_len, random_state=1)
	pos_mask = numpy.random.RandomState(2).random_sample(len(pos_seqs)) < 0.5
	pos_offsets = [center if m else None for m in pos_mask]
	pos_seqs = _plant(pos_seqs, motif_str, pos_offsets)

	neg_seqs = _random_sequences(100, seq_len, random_state=3)
	neg_mask = numpy.random.RandomState(4).random_sample(len(neg_seqs)) < 0.05
	rng = numpy.random.RandomState(5)
	neg_offsets = [int(rng.choice([0, seq_len - w])) if m else None
		for m in neg_mask]
	neg_seqs = _plant(neg_seqs, motif_str, neg_offsets)

	pos_path, neg_path = tmp_path / "pos.fasta", tmp_path / "neg.fasta"
	_write_fasta(pos_path, pos_seqs)
	_write_fasta(neg_path, neg_seqs)

	pwm = _motif_from_str(motif_str)
	meme_path = tmp_path / "motif.meme"
	write_meme(str(meme_path), {"m1": pwm})

	threshold = 0.05
	real = _run_real_centrimo(tmp_path, pos_path, meme_path, threshold,
		neg_fasta=neg_path)
	assert len(real) == 1
	real = real[0]

	result = centrimo({"m1": pwm}, _seqs_to_onehot(pos_seqs),
		control_sequences=_seqs_to_onehot(neg_seqs), threshold=threshold,
		eps=EPS)
	row = result.iloc[0]

	assert row["n_sequences"] == int(real["total_sites"])
	assert row["n_control_sequences"] == int(real["neg_sites"])
	assert abs(row["n_matching_sequences"] - int(real["sites_in_bin"])) <= 1
	assert abs(row["n_control_matching_sequences"] -
		int(real["neg_sites_in_bin"])) <= 1

	# fisher_e_value corresponds to the reference binary's `fisher_adj_pvalue`
	# (which is *not* further corrected by motif count, unlike `E-value` --
	# see centrimo.py's docstring). Compare on a log scale with a generous
	# tolerance, since it's derived from small, boundary-sensitive counts.
	assert numpy.log10(max(row["fisher_e_value"], 1e-300)) == pytest.approx(
		numpy.log10(max(float(real["fisher_adj_pvalue"]), 1e-300)), abs=1.5)


def test_centrimo_matches_reference_optimize_score(tmp_path):
	seq_len, w = 101, 8
	motif_str = "ACGTGCAT"
	center = (seq_len - w) // 2

	seqs = _random_sequences(150, seq_len, random_state=6)
	n_planted = 50
	offsets = [center] * n_planted + [None] * (len(seqs) - n_planted)
	seqs = _plant(seqs, motif_str, offsets)

	fasta_path = tmp_path / "seqs.fasta"
	_write_fasta(fasta_path, seqs)

	pwm = _motif_from_str(motif_str)
	meme_path = tmp_path / "motif.meme"
	write_meme(str(meme_path), {"m1": pwm})

	# A loose enough nominal threshold (see the note in
	# test_centrimo_optimize_score_finds_stricter_threshold in
	# test_centrimo.py) that plain background noise gets admitted alongside
	# the planted exact matches, giving `--optimize-score` room to find a
	# meaningfully stricter, more enriched cutoff.
	threshold = 1.0
	real = _run_real_centrimo(tmp_path, fasta_path, meme_path, threshold,
		extra_args=["--optimize-score"])
	assert len(real) == 1
	real = real[0]

	result = centrimo({"m1": pwm}, _seqs_to_onehot(seqs), threshold=threshold,
		eps=EPS, optimize_score=True)
	row = result.iloc[0]

	assert row["best_window_width"] == float(real["bin_width"])
	assert abs(row["n_sequences"] - int(real["total_sites"])) <= 1
	assert numpy.log10(row["e_value"]) == pytest.approx(
		numpy.log10(float(real["E-value"])), abs=1.5)


def test_centrimo_matches_reference_separate_strands(tmp_path):
	# A non-palindromic motif, planted only as its reverse complement, so
	# the real binary's `-m1` (reverse-complement) row is the significant
	# one and `+m1` isn't reported at all (falls below the E-value cutoff).
	seq_len = 41
	motif_str = "ACGTGCA"
	rc_str = "TGCACGT"
	w = len(motif_str)
	center = (seq_len - w) // 2

	seqs = _random_sequences(40, seq_len, random_state=20)
	n_planted = 20
	offsets = [center] * n_planted + [None] * (len(seqs) - n_planted)
	seqs = _plant(seqs, rc_str, offsets)

	fasta_path = tmp_path / "seqs.fasta"
	_write_fasta(fasta_path, seqs)

	pwm = _motif_from_str(motif_str)
	meme_path = tmp_path / "motif.meme"
	write_meme(str(meme_path), {"m1": pwm})

	threshold = 0.01
	real = _run_real_centrimo(tmp_path, fasta_path, meme_path, threshold,
		extra_args=["--sep"])
	real_by_id = {row["motif_id"].strip("'"): row for row in real}
	assert "-m1" in real_by_id

	result = centrimo({"m1": pwm}, _seqs_to_onehot(seqs), threshold=threshold,
		eps=EPS, separate_strands=True)
	assert list(result["motif_name"]) == ["m1", "m1-rc"]

	fwd_row = result[result["motif_name"] == "m1"].iloc[0]
	rc_row = result[result["motif_name"] == "m1-rc"].iloc[0]
	real_row = real_by_id["-m1"]

	# The forward row is just background noise -- confirm ours agrees it's
	# nowhere near as significant as the real reverse-complement row, even
	# though we always report both rows regardless of significance (ours
	# also caps e_value at 1.0, unlike the reference binary's E-value,
	# which can exceed 1 -- so compare orders of magnitude, not a raw >).
	assert fwd_row["e_value"] > 100 * rc_row["e_value"]

	assert rc_row["n_sequences"] == int(real_row["total_sites"])
	assert abs(rc_row["n_matching_sequences"] - int(real_row["sites_in_bin"])) <= 1
	assert rc_row["best_window_width"] == float(real_row["bin_width"])
	assert numpy.log10(rc_row["p_value"]) == pytest.approx(
		numpy.log10(float(real_row["p-value"])), abs=1.5)
	assert numpy.log10(rc_row["e_value"]) == pytest.approx(
		numpy.log10(float(real_row["E-value"])), abs=1.5)


def test_centrimo_matches_reference_flip_is_noop(tmp_path):
	# `--flip` is purely presentational in the reference binary too: confirm
	# its TSV output is byte-for-byte identical with and without the flag
	# (matching this package's own documented flip=True/False equivalence
	# on every reported statistic).
	seq_len = 41
	motif_str = "ACGTGCA"
	rc_str = "TGCACGT"
	w = len(motif_str)
	center = (seq_len - w) // 2

	seqs = _random_sequences(40, seq_len, random_state=21)
	n_planted = 20
	offsets = [center] * n_planted + [None] * (len(seqs) - n_planted)
	seqs = _plant(seqs, rc_str, offsets)

	fasta_path = tmp_path / "seqs.fasta"
	_write_fasta(fasta_path, seqs)

	pwm = _motif_from_str(motif_str)
	meme_path = tmp_path / "motif.meme"
	write_meme(str(meme_path), {"m1": pwm})

	threshold = 0.01
	real_noflip = _run_real_centrimo(tmp_path, fasta_path, meme_path,
		threshold, extra_args=["--sep"], tag="out_noflip")
	real_flip = _run_real_centrimo(tmp_path, fasta_path, meme_path,
		threshold, extra_args=["--sep", "--flip"], tag="out_flip")
	assert real_noflip == real_flip

	no_flip = centrimo({"m1": pwm}, _seqs_to_onehot(seqs), threshold=threshold,
		eps=EPS, separate_strands=True)
	flipped = centrimo({"m1": pwm}, _seqs_to_onehot(seqs), threshold=threshold,
		eps=EPS, separate_strands=True, flip=True)
	pandas.testing.assert_frame_equal(no_flip, flipped)

	rc_row = no_flip[no_flip["motif_name"] == "m1-rc"].iloc[0]
	real_row = real_noflip[0]
	assert numpy.log10(rc_row["e_value"]) == pytest.approx(
		numpy.log10(float(real_row["E-value"])), abs=1.5)
