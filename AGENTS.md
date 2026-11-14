# Repository guidance

These rules apply to every agent (Claude, Codex) working in this repository.

## Where things go

- Library code: `tvdcn/` (C++/CUDA sources in `tvdcn/csrc`, Python in
  `tvdcn/ops`, `tvdcn/extension.py`).
- Unit and regression tests: `tests/` (pytest). Runnable examples:
  `examples/`. User documentation: `docs/`. Packaging and maintenance
  scripts: `tools/`.
- Disposable scripts, experiments, benchmark output and logs: `develop/`
  (gitignored). Never commit them.
- Agent handoffs, plans and audit notes: `.agent-notes/` (ignored). Read
  `.agent-notes/resume.md` first; it holds the current state and the next
  step. Finished notes go to `.agent-notes/archive/`, generated logs to
  `.agent-notes/artifacts/`.
- Machine names, accounts and private paths stay out of tracked files. The
  lab environment is described in `.agent-notes/lab_workspace.md`; if it is
  absent, ask the maintainer before any remote build or test.

## Native code

- Target the PyTorch stable ABI (`torch/csrc/stable`, `torch/headeronly`)
  with `TORCH_TARGET_VERSION` set in `setup.py`. Do not include ATen, c10 or
  `torch/extension.h` headers in extension sources.
  Ops with no stable wrapper go through `torch_call_dispatcher` shims in
  `tvdcn/csrc/ops/StableABICompat.h`.
- Layers follow torchvision: schemas in `csrc/ops/*.cpp`, kernels in
  `csrc/ops/cpu` and `csrc/ops/cuda`; autograd, autocast and fake (meta)
  registrations in Python via `torch.library`.
- Preserve existing operator names and schemas (`tvdcn::deform_conv2d`,
  `tvdcn::_deform_conv2d_backward`, ...) so serialized graphs keep working.
- Ops must work with `torch.compile` and `torch.export`; check with
  `torch.library.opcheck`.

## Validation

- A change to kernels must keep parity with the golden fixtures in
  `tests/data/` (generated from the pre-port 1.1.0 build) and pass float64
  `gradcheck` on CPU and CUDA.
- Distinguish source inspection from executed tests. For CUDA results record
  the git revision, Python/torch/CUDA/nvcc/compiler versions, GPU and
  command. Never describe an unbuilt extension as CUDA-tested.
- CUDA builds and tests may run locally or on the authorized lab server (see
  the private notes). Do not upgrade shared environments, drivers or
  toolkits; the maintainer has no sudo on lab hosts.
- Keep tests focused on distinct behavior; reserve full CUDA grids for
  changes to shared kernels.

## Git and release

- Stage explicit paths; never `git add -A`. Commit only when asked.
- Publishing to PyPI happens only from the manual publish workflow, with the
  maintainer's explicit go-ahead.
