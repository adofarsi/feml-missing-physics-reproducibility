# Patch manifest

Each `.patch` here is plain `git diff` output and applies cleanly at the recorded
**base commit** of a fresh clone of the corresponding public repository:

```sh
git clone <repo> && cd <repo> && git checkout <base-commit>
git apply --check /path/to/<name>.patch   # dry run
git apply         /path/to/<name>.patch
```

These patches add the automatic-differentiation support for machine-learned
(base-form / external) operators used throughout the paper. They are being merged
upstream; until then, they reproduce the exact code used to run the experiments.

---

## firedrake.patch
- **Repository:** https://github.com/firedrakeproject/firedrake
- **Base commit:** `815bc4b1eba1532ea3cac90d881e7cd66f2788bd`
  — a **permanent ancestor on `master`** ("Ensemble reduced functional (#3500)", 2024-05-02).
  This is the merge-base of the AD-support work and `master`.
- **Files changed** (the full automatic-differentiation support for machine-learned
  base-form operators, layered on top of master):
  - `firedrake/ml/pytorch/ml_operator.py`, `firedrake/external_operators/ml_operator.py`
    — the ML operator itself (PyTorch coupling / external base-form operator).
  - `firedrake/assemble.py`, `firedrake/interpolation.py`,
    `firedrake/adjoint_utils/blocks/function.py` — assembly, interpolation, and adjoint paths.
  - `tests/regression/test_assemble_baseform.py`, `.github/workflows/build.yml` — tests and CI.

## ufl.patch
- **Repository:** https://github.com/firedrakeproject/ufl
- **Base commit:** `c1a8afb10bbde3c345d61a78b7f249f039dc6690` (master)
- **Files changed:**
  - `ufl/core/base_form_operator.py`, `ufl/core/interpolate.py`,
    `ufl/algorithms/replace.py`, `ufl/action.py`, `ufl/adjoint.py`, `ufl/form.py`
    — symbolic support and adjoints for external/ML base-form operators.
  - `test/test_external_operator.py` — corresponding tests.

## pyadjoint.patch
- **Repository:** https://github.com/dolfin-adjoint/pyadjoint
- **Base commit:** `92121af97711be88ff6d7994c0c7469129b2fb25` (master)
- **Files changed:**
  - `pyadjoint/block_variable.py` — `BlockVariable.add_adj_output`: fall back to a
    Riesz (`l2`) representation when accumulating adjoint contributions whose types
    do not add directly (as arises for ML-operator outputs).
