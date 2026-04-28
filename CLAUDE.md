# Project Rules: MulModSeg_2024 Bone Tumor Segmentation

## Git Versioning Rules

### Branch Strategy
- Main branch: `main` — stable, reviewed code only
- Experiment branches: `exp/<milestone>-<description>` (e.g., `exp/stage0-quick-wins`, `exp/m2b-freeze-strategy`)
- Each milestone gets its own branch, merged to `main` after results are confirmed

### Commit Rules
- **One commit per code change**: every implementation step (new utility, arg addition, dataloader fix) = one atomic commit
- **One commit per experiment config**: each new experiment configuration change = separate commit
- **Commit message format**:
  ```
  [M<milestone>] <ExpID>: <what changed>

  - bullet: specific change 1
  - bullet: specific change 2
  ```
  Example: `[M0] M0_boundary_dice: add boundary_dice_weight arg to train.py`
- Never amend published commits (force-push forbidden)
- Never skip pre-commit hooks (`--no-verify` forbidden)

### Tagging
- Tag successful experiment checkpoints: `exp/<ExpID>-best` (e.g., `exp/M0_boundary_dice-best`)
- Tag milestone completions: `milestone/<M-id>-complete` (e.g., `milestone/M0-complete`)

### What Goes in a Commit
- Code changes: model modifications, new utilities, dataloader changes
- Config changes: `configs/experiments.yaml` entries for new experiments
- Split files: `splits/*.json` when data splits change
- Experiment plan updates: `refine-log/EXPERIMENT_PLAN.md` and `refine-log/EXPERIMENT_TRACKER.md`

### What Does NOT Go in a Commit
- Model checkpoint files (`.pth`) — excluded via `.gitignore`
- Large pretrained weights — excluded via `.gitignore`
- Dataset files — excluded via `.gitignore`
- TensorBoard logs — excluded via `.gitignore`

## Run Rules

### Working Directory
Always run `train.py` from the project root (`/work/projhighcv/hzl/bone_tumor/`):
```bash
cd /work/projhighcv/hzl/bone_tumor
python MulModSeg_2024/train.py [args]
```
`train.py` adds the project root to `sys.path` automatically.

### Experiment Naming
Use `--log_name <ExpID>` to match the experiment tracker (e.g., `--log_name M0_boundary_dice`).

### Data Split
Use `--fold <0-4>` with `splits/fold5_splits.json` for reproducible 5-fold experiments.
Default (no `--fold`) uses the ratio-based split from the dataloader (legacy).

### GPU Assignment
- GPU 0: odd-numbered experiments
- GPU 1: even-numbered experiments
- Parallel experiments: one per GPU using `--device 0` / `--device 1`

## Bad Sample Policy
If a sample causes shape mismatch or other data errors:
1. Log the patient ID and error to `refine-log/bad_samples.json`
2. Add the ID to the `drop_list` in `splits/fold5_splits.json`
3. Update `refine-log/EXPERIMENT_PLAN.md` with the bad sample note
4. Never silently skip — always log

## Pretrained Weight Policy
- SSL pretrained weights: `MulModSeg_2024/pretrained/ssl_pretrained_weights.pth` (tracked in git as untracked — do not add to git)
- BTCV/MONAI weights: download on demand, store in `MulModSeg_2024/pretrained/` (not committed)
- BiomedCLIP weights: HuggingFace cache (not committed)
- Text embeddings: `MulModSeg_2024/text_embedding/` (committed if small, excluded if large)
