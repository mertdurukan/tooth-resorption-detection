# Contributing

This is an MSc research repo, but PRs that tighten the science or the
engineering are welcome.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Local checks (must pass before opening a PR)

```bash
ruff check src scripts tests
ruff format --check src scripts tests
mypy src
pytest --cov=tooth_resorption
```

CI runs the same four steps on Python 3.10 and 3.11.

## Conventions

- **Code in English, user-facing docs may stay Turkish where intentional.**
  Class names ship as ASCII (`temasli` / `bagimsiz` / `rezorpsiyon`) for
  cross-platform safety; the Turkish accented forms appear only in
  human-readable documentation.
- **Type hints are required on every public function in `src/tooth_resorption/`.**
  Mypy is configured in strict mode against `src/`.
- **No bare `print()` in `src/tooth_resorption/`.** Use
  `tooth_resorption.logging_utils.get_logger(__name__)`. `print()` is
  permitted in `scripts/` because those are CLI entry-points.
- **Paths via `pathlib.Path`.** Avoid `os.path.join`.
- **No mutable default arguments.** Use `None` sentinels.
- **f-strings** for all formatting. No `%` or `.format`.
- **Tests live under `tests/`** and follow the `test_<module>.py` naming.
  Use the shared fixtures in `conftest.py`. Mark slow / network-bound tests
  with `@pytest.mark.slow`.

## Data handling

- **Never commit patient data, model weights or large binaries.** The
  `.gitignore` already excludes `*.h5/.pt/.pth/.onnx`, `models/`,
  `data/raw/`, `data/processed/`, `data/synthetic/` and the legacy
  Turkish folder `20 lik diş rezorpsiyon/`.
- **Never modify `results/metrics.json` to inflate numbers.** Those values
  come from the original MSc thesis evaluation runs and are the only
  metrics cleared for publication.

## Commit / PR style

- Small, focused commits with imperative subject lines (`Add stratified
  split for val loader`).
- PR description should state **what** changed, **why**, and how it was
  verified locally (`pytest -q`, eyeballed confusion matrix, etc.).
