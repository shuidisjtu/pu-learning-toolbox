# Development and Compatibility Policy

## 1. Supported environments

| Item | Current policy |
|---|---|
| Python | 3.10, 3.11, 3.12 |
| Recommended development version | 3.11 (`.python-version`) |
| Core dependencies | numpy, scipy, pandas, scikit-learn |
| Deep learning | `torch` extra |
| Research extensions | `research` extra: densratio, torch, torchvision, lightning, tqdm, sentence-transformers |
| CI platforms | GitHub Actions `ubuntu-latest` / `windows-latest` / `macos-latest` |
| Build backend | Hatchling |

Python classifiers, the Ruff target, the CI matrix, and this document must stay consistent.
Before adding a Python version, run the full test suite and a wheel-installation smoke test.
Record removal of version support in the release notes.

## 2. Installation combinations

```bash
pip install -e .                 # Core numpy/sklearn functionality
pip install -e ".[torch]"       # nnPU, Dist-PU, and deep methods
pip install -e ".[research]"    # uLSIF, image, text, and Lightning research environment
pip install -e ".[all]"         # All runtime features, excluding development tools
pip install -e ".[dev]"         # Tests, lint, and build tools
```

Optional dependencies must have guarded or delayed imports. With only core dependencies installed,
`import pu_toolbox` must work. A training path requiring torch, or the uLSIF comparison in the
PUSB benchmark, must provide a clear installation hint when its dependency is missing.

## 3. Dependency sources

`pyproject.toml` defines installation requirements. As a library, the project uses minimum-version
constraints. CI installs from the committed `uv.lock`, shared with local development for
deterministic, cross-platform resolution. The weekly workflow runs `uv sync --upgrade` to detect
upstream compatibility problems; this updates only its temporary checkout.
The lockfile replaced `requirements.txt`, which was removed on 2026-09-08.

**Base-import guarantee:** the package's own imports must work when optional dependencies are
unavailable. Blocking `torch`, `torchvision`, and `streamlit` has been verified to leave
`import pu_toolbox` working. There are no unconditional module-level torch imports.
The two module-scope torch import statements are guarded:

- `pu_toolbox/estimators/deep/vision.py` imports `nn` inside a `try` block with an
  `except ImportError` fallback.
- `pu_toolbox/core/device.py` imports torch under `if TYPE_CHECKING:`, which is not executed
  at runtime.

This guarantee does not require base imports to leave installed optional dependencies unloaded.
When torch is installed, the root package's re-exports reach the guarded import and load torch;
`torch/hub.py` also loads tqdm. The experiment layer and its research dependencies
(`densratio`, `lightning`, `sentence_transformers`) remain unloaded. A stronger guarantee
would require lazy root-package exports and is a separate architectural change.

## 4. CI structure

CI separates these responsibilities:

1. **Tests (fast PR suite):** an explicit Python 3.10/3.11/3.12 matrix on Ubuntu, Windows, and macOS
   (3 × 3). It installs the dev and torch extras, including torchvision, so vision unit tests
   actually run. It executes tests excluding `slow` and `e2e` (unit and integration).
2. **Static quality gates:** Python 3.11 runs the static checks. The authoritative commands are the
   Quality gates block under Development in [README](../../README.md) and the `quality` job in
   `.github/workflows/tests.yml`; use those lists to avoid maintaining a divergent copy.
3. **UI tests:** a dedicated Python 3.11 job installs the UI extra and exercises the Streamlit
   application.
4. **Build and install wheel:** build the sdist and wheel, install the wheel in an isolated
   environment, and verify the version, diagnostics import, nonempty registry, CLI entry points,
   and optional UI entry point from outside the checkout.
5. **Weekly slow suite:** the workflow named Nightly runs every Monday at 03:23 UTC on the same
   3 × 3 matrix. It executes `-m "slow or e2e"` with the same test dependencies, including
   torchvision.

Explicit interpreter assertions prevent `.python-version` from accidentally overriding the
CI matrix.

**Test-selection scope:** the `quality` job runs static gates only. The test jobs run
`pytest tests/ -m "not slow and not e2e"` for pushes and pull requests targeting `main`
(`.github/workflows/tests.yml`). There are no path filters: changes to any subpackage, including
`experiment/`, run the entire fast suite. The `slow or e2e` suite runs weekly in
`.github/workflows/nightly.yml`. Survey changes therefore run the full fast regression suite;
they do not automatically trigger the full slow suite.

## 5. Distribution contents

Hatchling configuration lives in `pyproject.toml`:

- The wheel contains `pu_toolbox`, distribution metadata, and the
  `.claude/skills/pu-workflow` resources explicitly included by the build configuration.
- The sdist includes source, tests, benchmarks, documentation, examples, scripts, Skill resources,
  and project configuration.
- Builds do not depend on `MANIFEST.in`.

Before a release, run:

```bash
uv build
```

Install the generated wheel in a clean environment and test it outside the checkout so local
source files cannot conceal missing distribution files.

## 6. Known limits

- Ordinary torch environments verify the basic interfaces of deep methods. Paper-scale GPU,
  CUDA, and historical dependency environments require separate environment locks.
- Full DGPU experiments require an external EDM/diffusion-generator backend.
- Clean-room implementations and paper-like results do not establish reproduction in the
  official historical environment.
