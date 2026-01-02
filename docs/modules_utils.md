# Module: utils

## 1) Purpose
- General utility functions (math, tensor helpers, logging) shared across modules.

## 2) Files
- `modules/utils.py` — Misc helper functions.

## 3) Public APIs
- Helper functions (see file) for geometry, tensor ops, and convenience utilities.
- Called by multiple modules (`target_generator`, `metrics`, encoders).

## 4) Data Flow & Tensor Shapes
- Varies by helper; typically operates on torch tensors/NumPy arrays and returns same-shape outputs.

## 5) Configuration
- None; stateless helpers.

## 6) Training vs Inference Behavior
- No difference; pure functions.

## 7) Troubleshooting
- dtype/device issues: ensure tensors on correct device before calling.
- Unexpected inplace modifications: check helper behavior if mutating inputs.
- Import errors: verify correct module path (`modules.utils`).

## 8) Alignment to Proposal
- Utility support; not a core proposal item. Status: **Fully implemented** for current needs.

## 9) Minimal-change Extension Guide
- Add new helpers at end of `utils.py`; keep function names stable; avoid side effects.***

