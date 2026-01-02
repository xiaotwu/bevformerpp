# Training and Notebook Usage

## train.py Structure
- **Argument parsing**: Temporal method, fusion flag, epochs, batch size, LR, curriculum stage parameters (heatmap_mode, Gaussian overlap/radius, pos_weight per stage).
- **Datasets and loaders**: `NuScenesDataset` with `create_collate_fn` / `create_fusion_collate_fn`; loaders recreated when curriculum stage switches.
- **Models**:
  - Camera-only: `EnhancedBEVFormer` backbone + `BEVHead`.
  - Fusion: `BEVFusionModel` for LiDAR + camera.
- **Loss**: `modules.head.DetectionLoss` (BCE with logits) recreated per stage with stage-specific `pos_weight`.
- **Curriculum**:
  - Stage 1: `heatmap_mode=hard_center`, strong pos_weight.
  - Stage 2: `heatmap_mode=gaussian`, relaxed Gaussian params and pos_weight.
  - Logs stage parameters on each switch.
- **Training loop**: forward → loss → backward → grad clip → optimizer.step; validation per epoch; test after training.

## Notebook (main.ipynb) Guidance
- The notebook must **call into `train.py`** (import functions or invoke as a script).
- Do not duplicate training loops; reuse the same arguments/curriculum.
- Recommended snippets:
  - `%run train.py --epochs 2 --stage1_epochs 1`
  - Or `import train` and call the appropriate train function with parsed args.
- Use notebook only for:
  - Visualizing heatmaps/BEV outputs.
  - Inspecting predictions vs GT.
  - Running small overfit/debug runs that still use `train.py` logic.

## Checkpointing and Eval
- Evaluation occurs after each epoch on the val loader; a final test run is executed after training.
- Checkpointing: follow existing `train.py` hooks if present; otherwise extend minimally by saving model/optimizer states after epochs (keep consistent with training arguments).


