# Repository Documentation

This folder documents the BEV-based multi-modal perception system. Start here for structure and navigation, then drill down into the module pages.

## How to Read This Repo
- Begin with `architecture_overview.md` for end-to-end data flow and BEV grid assumptions.
- Review `config_reference.md` to see which YAML/loader files control data, model, and training hyperparameters.
- Check `training_and_notebook.md` for how to run training (only `train.py`) and how notebooks should call into it.
- For component details, open the corresponding `docs/modules_<name>.md`.

## Modules by Pipeline Stage
- **Data**: `nuscenes_dataset` (loading, collate, target generation wrapper), `preprocessing` (augment/normalize), `target_generator` (heatmap/reg targets).
- **LiDAR BEV**: `lidar_encoder` (PointPillars-style voxel/pillar encoding).
- **Camera BEV**: `camera_encoder`, `bevformer`, `attention`, `temporal_attention`.
- **BEV Fusion**: `fusion`, `bev_fusion_model`, `neck`, `fusion` utilities.
- **Temporal**: `temporal_attention`, `memory_bank`, `convrnn`, `mc_convrnn`.
- **Detection Head**: `head`, `losses`, `metrics`.
- **Backbone/Shared**: `backbone`, `neck`, `utils`, `data_structures`.
- **Training/Utils**: `train.py`, `config_loader`, `base/eval/train YAMLs`, `metrics`.

## Key Entry Points
- `train.py` is the **only** training entrypoint.
- `main.ipynb` is for visualization/debugging and must call into `train.py` logic.


