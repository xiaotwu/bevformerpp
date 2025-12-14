# Requirements Document

## Introduction

This document specifies the requirements for a modular Bird's-Eye View (BEV) perception system that integrates LiDAR point clouds and multi-view camera images for 3D object detection. The system combines geometric features from LiDAR (via PointPillars) with semantic features from cameras (via BEVFormer), performs spatial fusion in BEV space, and applies temporal aggregation using both transformer-based attention and motion-compensated ConvRNN approaches. The target dataset is nuScenes v1.0-mini.

## Glossary

- **BEV**: Bird's-Eye View - a top-down 2D representation of 3D space
- **PointPillars**: A LiDAR-based 3D object detection method that converts point clouds to pillars
- **BEVFormer**: A transformer-based method for generating BEV features from multi-view camera images
- **Spatial Fusion Module**: A component that combines LiDAR and camera BEV features using cross-attention
- **Temporal Module**: A component that aggregates features across time frames
- **MC-ConvRNN**: Motion-Compensated Convolutional Recurrent Neural Network for temporal fusion
- **Detection Head**: The final network component that predicts 3D bounding boxes
- **nuScenes Dataset**: A public autonomous driving dataset with LiDAR and camera data
- **Ego-motion**: The movement of the vehicle/sensor platform between frames
- **System**: The complete BEV fusion perception pipeline

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to process LiDAR point clouds into BEV features, so that I can obtain geometric representations for 3D object detection.

#### Acceptance Criteria

1. WHEN the System receives a point cloud with coordinates (x, y, z, r), THE System SHALL convert the point cloud into pillar representations within a defined BEV grid
2. WHEN pillars are created, THE System SHALL encode each pillar using a PointNet-style encoder to extract per-pillar features
3. WHEN pillar features are encoded, THE System SHALL scatter them to a 2D BEV grid of dimensions H × W × C1
4. WHEN the BEV grid is populated, THE System SHALL process it through a 2D CNN backbone to produce the final LiDAR BEV feature map
5. THE System SHALL output LiDAR BEV features with shape (B, C1, H, W) where B is batch size, C1 ≈ 64, and H, W ≈ 200

### Requirement 2

**User Story:** As a researcher, I want to process multi-view camera images into BEV features, so that I can obtain semantic representations for 3D object detection.

#### Acceptance Criteria

1. WHEN the System receives six multi-view RGB images with camera intrinsics and extrinsics, THE System SHALL extract image features using a backbone network
2. WHEN image features are extracted, THE System SHALL apply spatial cross-attention between BEV queries and image features to project them into BEV space
3. THE System SHALL generate camera BEV features with shape (B, C2, H, W) where C2 ≈ 128-256
4. WHEN camera calibration parameters are provided, THE System SHALL use them to correctly map image features to BEV coordinates
5. THE System SHALL maintain spatial alignment between camera BEV and LiDAR BEV representations

### Requirement 3

**User Story:** As a researcher, I want to fuse LiDAR and camera BEV features, so that I can combine geometric and semantic information for improved detection.

#### Acceptance Criteria

1. WHEN the System receives LiDAR BEV features F_lidar and camera BEV features F_cam, THE System SHALL apply cross-attention fusion between them
2. WHEN fusion is performed, THE System SHALL produce unified BEV features F_fused with shape (B, C3, H, W) where C3 ≈ 256
3. THE System SHALL ensure spatial alignment between LiDAR and camera BEV grids before fusion
4. THE System SHALL implement lightweight fusion to minimize computational latency
5. WHEN fusion completes, THE System SHALL output features that preserve both geometric and semantic information

### Requirement 4

**User Story:** As a researcher, I want to aggregate temporal information using transformer-based attention, so that I can improve detection consistency across frames.

#### Acceptance Criteria

1. WHEN the System receives a sequence of T past BEV features (T = 3-5), THE System SHALL align them using ego-motion transformations
2. WHEN features are aligned, THE System SHALL apply temporal self-attention across the sequence
3. WHEN temporal attention is computed, THE System SHALL apply adaptive confidence weighting through temporal gating
4. THE System SHALL update current features using residual connections with temporally aggregated features
5. THE System SHALL output temporally enhanced BEV features F_temp_attn with shape (B, C, H, W)

### Requirement 5

**User Story:** As a researcher, I want to aggregate temporal information using motion-compensated ConvRNN, so that I can achieve stable long-range temporal reasoning.

#### Acceptance Criteria

1. WHEN the System receives previous BEV feature F_{t-1} and current feature F_t with ego-motion transform, THE System SHALL warp F_{t-1} using the ego-motion transformation
2. WHEN warping is complete, THE System SHALL compute a dynamic residual motion field for fine-grained alignment
3. WHEN motion alignment is done, THE System SHALL apply visibility gating using a visibility mask M_t
4. WHEN visibility gating is applied, THE System SHALL fuse features using ConvGRU recurrent fusion
5. THE System SHALL output motion-compensated temporally enriched BEV features F'_t with shape (B, C, H, W)

### Requirement 6

**User Story:** As a researcher, I want to predict 3D bounding boxes from BEV features, so that I can detect objects in 3D space.

#### Acceptance Criteria

1. WHEN the System receives final BEV features, THE System SHALL apply dense 2D CNN prediction over the BEV grid
2. WHEN predictions are made, THE System SHALL output classification scores for object categories
3. WHEN predictions are made, THE System SHALL output regression parameters for 3D bounding boxes including center, size, and yaw angle
4. WHEN raw predictions are generated, THE System SHALL apply Non-Maximum Suppression (NMS) post-processing
5. THE System SHALL output a list of 3D bounding boxes with confidence scores

### Requirement 7

**User Story:** As a researcher, I want to load and preprocess nuScenes dataset, so that I can train and evaluate the perception system.

#### Acceptance Criteria

1. WHEN the System initializes the dataset, THE System SHALL load nuScenes v1.0-mini data from the specified directory
2. WHEN loading samples, THE System SHALL extract synchronized LiDAR point clouds and six camera images per frame
3. WHEN loading samples, THE System SHALL load camera calibration matrices (intrinsics and extrinsics)
4. WHEN loading samples, THE System SHALL load ego-motion transformations between consecutive frames
5. WHEN loading samples, THE System SHALL load ground truth 3D bounding box annotations for training and evaluation

### Requirement 8

**User Story:** As a researcher, I want to train the perception system, so that I can optimize model parameters for 3D object detection.

#### Acceptance Criteria

1. WHEN training begins, THE System SHALL initialize all network components with appropriate random weights or pretrained weights
2. WHEN processing training batches, THE System SHALL compute detection losses including classification loss and regression loss
3. WHEN losses are computed, THE System SHALL backpropagate gradients and update model parameters using an optimizer
4. WHEN training progresses, THE System SHALL log training metrics including loss values and learning rate
5. WHEN training completes an epoch, THE System SHALL save model checkpoints to disk

### Requirement 9

**User Story:** As a researcher, I want to evaluate the perception system, so that I can measure detection performance on validation data.

#### Acceptance Criteria

1. WHEN evaluation begins, THE System SHALL load a trained model checkpoint from disk
2. WHEN processing validation samples, THE System SHALL run inference without gradient computation
3. WHEN predictions are generated, THE System SHALL compute evaluation metrics including Average Precision (AP) and nuScenes Detection Score (NDS)
4. WHEN evaluation completes, THE System SHALL output metric results in a readable format
5. THE System SHALL support visualization of predictions overlaid on input data for qualitative assessment

### Requirement 10

**User Story:** As a researcher, I want to configure system hyperparameters, so that I can experiment with different architectural choices and training settings.

#### Acceptance Criteria

1. THE System SHALL support configuration of BEV grid dimensions (H, W) and resolution
2. THE System SHALL support configuration of feature dimensions (C1, C2, C3) for each module
3. THE System SHALL support configuration of temporal sequence length T
4. THE System SHALL support configuration of training hyperparameters including learning rate, batch size, and number of epochs
5. THE System SHALL load configuration from structured files (e.g., YAML or Python config files)
