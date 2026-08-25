# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`memelite` (repo name `memesuite-lite`) is a from-scratch Python re-implementation of core [MEME suite](https://meme-suite.org/meme/) algorithms — Tomtom (motif-vs-motif similarity p-values) and FIMO (motif-vs-sequence scanning) — without a PyTorch dependency. The implementations are numba-jitted for speed and designed to scale to millions of queries/targets on consumer hardware. It ships as a pip package (`pip install memelite`) plus a `ttl` command-line tool.

## Commands

This project uses `uv` for dependency management.

```bash
uv sync                    # install dependencies (including dev group)
uv run pytest              # run the full test suite
uv run pytest tests/test_tomtom.py                    # run one test file
uv run pytest tests/test_tomtom.py::test_tomtom_basic  # run one test
uv run pytest -m "not cmd" # skip tests that shell out to the `ttl` CLI
uv run pytest -m cmd       # run only the `ttl` CLI subprocess tests
uv run pytest -m meme_ref  # run only the real-MEME-suite equivalence tests
```

The `cmd`-marked tests (see `tests/test_cli.py`) invoke the installed `ttl` console script via `os.system`, so the package must be installed (e.g. `uv sync` / editable install) for them to pass — they aren't pure in-process unit tests.

The `meme_ref`-marked tests (e.g. `tests/test_centrimo_equivalence.py`) shell out to a real, separately-installed MEME suite binary (e.g. `conda create -n meme-ref -c bioconda -c conda-forge meme=5.5.9`) to check this package's output against the reference implementation. They skip automatically (not via the `not cmd`-style marker exclusion) if the relevant binary isn't found on PATH, so `uv run pytest` always runs cleanly without one installed.

There is no configured linter/formatter in this repo (flake8 is invoked in CI but with all checks commented out).

## Architecture

Everything lives in `memelite/`. The public API (`memelite/__init__.py`) exposes exactly three functions: `fimo`, `tomtom`, `symmetric_tomtom`.

- **`tomtom.py`** — the core Tomtom algorithm. Only the "complete score" variant is implemented (the "incomplete score" from the original MEME suite is intentionally omitted as inferior). The algorithm has two conceptual phases, split across several `@njit` helpers: (1) compute alignment scores between every query/target column pair and discretize them into bins (`_integer_distances_and_histogram`), using an approximate binned-median routine (`_binned_median`) rather than exact sorting; (2) convert the score histograms into a background distribution and derive p-values via dynamic programming over all possible overhangs (`_p_value_backgrounds`, `_p_values`). `_merge_rc_results` combines forward and reverse-complement strand results. The public `tomtom()` function wraps `_tomtom()` and supports an `n_nearest` mode that returns only the top-k target matches per query (with an index matrix) instead of the full dense `n_queries x n_targets` matrix, to keep memory bounded at scale. `n_target_bins` controls optional approximate hashing that merges similar target columns to trade accuracy for speed.
- **`symmetric_tomtom.py`** — a specialized path for comparing one set of motifs against itself (`symmetric_tomtom(Xs, ...)`). It reuses several private helpers imported directly from `tomtom.py` (`_binned_median`, `_pairwise_max`, `_merge_rc_results`, `_integer_distances_and_histogram`, `_p_values`) but has its own `_p_value_backgrounds`/`_tomtom` tailored to the symmetric case (e.g. avoiding redundant self-comparisons). Keep the shared helpers in sync when editing either file.
- **`fimo.py`** — the FIMO algorithm: scan PWMs across one-hot encoded sequences like a convolution, then map raw scores to p-values using an exact background distribution derived from the PWM itself via dynamic programming (`_pwm_to_mapping`, `_all_pwm_to_mapping`), using `logaddexp2` for numerically stable log-space accumulation. `_fast_hits` does the actual scanning across all sequences/motifs; `fimo()` is the public entry point and returns one pandas `DataFrame` of hits per motif. FIMO assumes long target sequences (overhangs don't matter), which is the opposite assumption from Tomtom.
- **`io.py`** — `read_meme`/`write_meme` for parsing and writing MEME-formatted PWM files (dict of motif name -> `(alphabet_size, length)` numpy array).
- **`utils.py`** — `one_hot_encode` (string -> one-hot array) and `characters` (PWM/one-hot array -> consensus string), the two conversions used to move between MEME/FASTA text formats and the numeric arrays the numba kernels operate on.
- **`cli.py`** — implements the `ttl` console script (registered in `pyproject.toml` under `[project.scripts]`). It supports two mutually exclusive modes: `_run_tomtom` (query is a MEME file or a raw sequence string, compared against a target MEME database — defaults to auto-downloading JASPAR2024 if `-t` isn't given) and `_run_annotate` (BED + FASTA input, annotates each coordinate with its best motif match). Also contains the alignment-display logic that upper/lower-cases matched vs. mismatched positions, accounting for which strand won.

### Data flow / conventions

- PWMs and one-hot sequences are numpy arrays shaped `(len(alphabet), length)` (the PyTorch convention — alphabet dimension first), even though this package has no PyTorch dependency; inputs may be passed in as torch tensors and are converted internally.
- Motif collections are typically Python dicts of `name -> array`; `read_meme` produces this shape directly.
- Numba `@njit(cache=True)` is used pervasively for the hot inner loops — when modifying these functions, keep signatures numba-compatible (no dynamic Python objects) and be aware that `cache=True` persists compiled artifacts, so a stale `__pycache__`/numba cache can mask changes during iteration.
- Reverse-complement handling is threaded through both `tomtom` (`reverse_complement=True` default, merged via `_merge_rc_results`) and the CLI's alignment renderer — changes to strand logic in one likely need matching changes in the other.
