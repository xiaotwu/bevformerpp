# Design Document

## Overview

This document describes the detailed design of a modular Bird's-Eye View (BEV) perception system for 3D object detection. The system integrates LiDAR point clouds and multi-view camera images through a multi-stage pipeline: (1) LiDAR BEV encoding via PointPillars, (2) Camera BEV encoding via BEVFormer, (3) Spatial fusion using cross-attention, (4) Temporal aggregation using both transformer-based attention and motion-compensated ConvRNN, and (5) 3D bounding box prediction via a detection head.

The design prioritizes modularity, allowing independent development and testing of each component, while maintaining end-to-end differentiability for joint training.

## Architecture

### System Pipeline

```
┌─────────────────┐
│  Point Cloud    │
│   (x,y,z,r)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│ LiDAR Encoder   │      │ Camera Encoder  │
│ (PointPillars)  │      │  (BEVFormer)    │
└────────┬────────┘      └────────┬────────┘
         │                        │
         │  F_lidar              │  F_cam
         │  (B,C1,H,W)           │  (B,C2,H,W)
         │                        │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Spatial Fusion │
         │ (Cross-Attn)   │
         └────────┬───────┘
                  │
                  │  F_fused (B,C3,H,W)
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌──────────────────┐
│ Temporal Attn   │  │  MC-ConvRNN      │
│ (Transformer)   │  │  (Recurrent)     │
└────────┬────────┘  └────────┬─────────┘
         │                    │
         │  F_temp_attn       │  F'_t
         │                    │
         └────────┬───────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Detection Head │
         │  (3D Boxes)    │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  3D Bounding   │
         │     Boxes      │
         └────────────────┘
```

### Module Hierarchy

- **BEVFusionModel** (top-level)
  - LiDARBEVEncoder
    - PillarFeatureNet
    - PointPillarsScatter
    - BackboneCNN
  - CameraBEVEncoder
    - ImageBackbone (ResNet50)
    - BEVFormerEncoder
      - SpatialCrossAttention
      - TemporalSelfAttention (optional)
  - SpatialFusionModule
    - CrossAttentionFusion
  - TemporalModule
    - TemporalSelfAttention (with memory bank)
    - MCConvRNN
      - EgoMotionWarping
      - MotionFieldEstimator
      - VisibilityGating
      - ConvGRU
  - DetectionHead
    - ClassificationHead
    - RegressionHead

## Components and Interfaces

### 1. LiDAR BEV Encoder

**Purpose:** Convert raw LiDAR point clouds into BEV feature representations.

**Input:**
- `points`: Tensor of shape `(N, 4)` where N is number of points, columns are (x, y, z, reflectance)
- `voxel_size`: Tuple `(vx, vy, vz)` defining pillar dimensions
- `point_cloud_range`: Tuple `(x_min, y_min, z_min, x_max, y_max, z_max)`

**Output:**
- `F_lidar`: Tensor of shape `(B, C1, H, W)` where C1=64, H=W=200

**Sub-components:**

#### 1.1 PillarFeatureNet
- Voxelizes point cloud into pillars
- Applies PointNet-style encoding per pillar
- Returns pillar features and coordinates

#### 1.2 PointPillarsScatter
- Scatters pillar features to dense BEV grid
- Handles empty pillars with zero-padding

#### 1.3 BackboneCNN
- 2D CNN with residual blocks
- Progressively refines BEV features
- Architecture: Conv2d layers with BatchNorm and ReLU

**Key Design Decisions:**
- Use sparse convolutions for efficiency if pillar occupancy is low
- Pillar height aggregation via max-pooling
- BEV grid resolution: 0.2m per pixel (configurable)

---

### 2. Camera BEV Encoder

**Purpose:** Convert multi-view camera images into BEV feature representations.

**Input:**
- `images`: Tensor of shape `(B, N_cam, 3, H_img, W_img)` where N_cam=6
- `intrinsics`: Tensor of shape `(B, N_cam, 3, 3)`
- `extrinsics`: Tensor of shape `(B, N_cam, 4, 4)`

**Output:**
- `F_cam`: Tensor of shape `(B, C2, H, W)` where C2=256

**Sub-components:**

#### 2.1 ImageBackbone
- ResNet50 pretrained on ImageNet
- Extracts multi-scale image features
- Returns feature pyramid

#### 2.2 BEVFormerEncoder
- Initializes learnable BEV queries `Q_bev` of shape `(H*W, C2)`
- Applies spatial cross-attention between BEV queries and image features
- Uses deformable attention for efficiency
- Projects image features to BEV space using camera geometry

**Spatial Cross-Attention Mechanism:**
```
For each BEV query at position (u, v):
  1. Compute 3D reference points via inverse projection
  2. Sample image features at projected 2D locations
  3. Apply multi-head attention: Q_bev attends to image features
  4. Aggregate across all camera views
```

**Key Design Decisions:**
- Use deformable attention to reduce computational cost
- Number of attention heads: 8
- Number of attention layers: 6
- BEV query positional encoding: 2D sine-cosine embeddings

---

### 3. Spatial Fusion Module

**Purpose:** Combine LiDAR and camera BEV features into unified representation.

**Input:**
- `F_lidar`: Tensor of shape `(B, C1, H, W)`
- `F_cam`: Tensor of shape `(B, C2, H, W)`

**Output:**
- `F_fused`: Tensor of shape `(B, C3, H, W)` where C3=256

**Fusion Strategy:**

#### Option A: Cross-Attention Fusion (Primary)
```python
# LiDAR attends to camera
Q_lidar = Linear(F_lidar)  # (B, H*W, C3)
K_cam = Linear(F_cam)      # (B, H*W, C3)
V_cam = Linear(F_cam)      # (B, H*W, C3)
Attn_out = MultiHeadAttention(Q_lidar, K_cam, V_cam)
F_fused = F_lidar + Attn_out  # Residual connection
```

#### Option B: Concatenation + Conv (Fallback)
```python
F_concat = Concat([F_lidar, F_cam], dim=1)  # (B, C1+C2, H, W)
F_fused = Conv2d(F_concat)  # (B, C3, H, W)
```

**Key Design Decisions:**
- Use cross-attention for better feature interaction
- Apply layer normalization before attention
- Use residual connections to preserve LiDAR geometric information
- Lightweight design: single attention layer

---

### 4. Temporal Module

**Purpose:** Aggregate information across time to improve temporal consistency.

The system implements two parallel temporal aggregation approaches:

#### 4.1 Temporal Self-Attention

**Input:**
- `F_fused_history`: List of T past fused BEV features
- `ego_motion_transforms`: List of T-1 SE(3) transforms

**Output:**
- `F_temp_attn`: Tensor of shape `(B, C, H, W)`

**Process:**
1. **Alignment:** Warp past features to current frame using ego-motion
2. **Temporal Self-Attention:** Apply multi-head self-attention across time
3. **Temporal Gating:** Compute confidence weights for each frame
4. **Residual Update:** Combine with current features

**Alignment Function:**
```python
def align_bev_features(F_past, ego_transform):
    # ego_transform: (B, 4, 4) SE(3) matrix
    # Generate sampling grid from ego-motion
    grid = generate_grid_from_transform(ego_transform, H, W)
    # Warp features using grid sampling
    F_aligned = grid_sample(F_past, grid, mode='bilinear')
    return F_aligned
```

**Temporal Gating:**
```python
# Compute confidence scores based on feature similarity
confidence = sigmoid(Conv2d(concat([F_current, F_aligned])))
F_gated = confidence * F_aligned
```

**Key Design Decisions:**
- Sequence length T = 3-5 frames
- Use deformable attention for efficiency
- Store features in memory bank (FIFO queue)
- Apply temporal dropout during training for robustness

---

#### 4.2 Motion-Compensated ConvRNN (MC-ConvRNN)

**Input:**
- `F_t`: Current fused BEV feature `(B, C, H, W)`
- `F_{t-1}`: Previous BEV feature `(B, C, H, W)`
- `H_{t-1}`: Previous hidden state `(B, C_h, H, W)`
- `ego_motion`: SE(3) transform `(B, 4, 4)`

**Output:**
- `F'_t`: Motion-compensated BEV feature `(B, C, H, W)`
- `H_t`: Updated hidden state `(B, C_h, H, W)`

**Process (5 steps):**

**Step 1: Ego-Motion Warping**
```python
F_warped = warp_bev_features(F_{t-1}, ego_motion)
H_warped = warp_bev_features(H_{t-1}, ego_motion)
```

**Step 2: Dynamic Residual Motion Field**
```python
# Estimate fine-grained motion beyond ego-motion
motion_field = MotionFieldEstimator(F_t, F_warped)  # (B, 2, H, W)
F_aligned = warp_with_flow(F_warped, motion_field)
```

**Step 3: Visibility Gating**
```python
# Compute visibility mask (1 = visible, 0 = occluded/out-of-view)
visibility_mask = compute_visibility_mask(ego_motion, H, W)
F_gated = visibility_mask * F_aligned
```

**Step 4: ConvGRU Fusion**
```python
# Standard ConvGRU update
z_t = sigmoid(Conv(concat([F_t, H_warped])))  # Update gate
r_t = sigmoid(Conv(concat([F_t, H_warped])))  # Reset gate
H_tilde = tanh(Conv(concat([F_t, r_t * H_warped])))
H_t = (1 - z_t) * H_warped + z_t * H_tilde
```

**Step 5: Output Projection**
```python
F'_t = Conv(concat([F_t, H_t]))
```

**Key Design Decisions:**
- Motion field estimator: lightweight CNN (3 conv layers)
- Visibility mask: computed from ego-motion and BEV bounds
- ConvGRU hidden dimension: C_h = 128
- Initialize H_0 with zeros for first frame

---

### 5. Detection Head

**Purpose:** Predict 3D bounding boxes from BEV features.

**Input:**
- `F_final`: Tensor of shape `(B, C, H, W)` (from temporal module)

**Output:**
- `cls_scores`: Tensor of shape `(B, N_cls, H, W)` where N_cls is number of classes
- `bbox_preds`: Tensor of shape `(B, 7, H, W)` encoding (x, y, z, w, l, h, yaw)

**Architecture:**
```python
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.cls_head = nn.Conv2d(256, num_classes, 1)
        self.reg_head = nn.Conv2d(256, 7, 1)  # (x, y, z, w, l, h, yaw)
    
    def forward(self, x):
        shared_feat = self.shared_conv(x)
        cls_scores = self.cls_head(shared_feat)
        bbox_preds = self.reg_head(shared_feat)
        return cls_scores, bbox_preds
```

**Post-Processing:**
1. Apply sigmoid to classification scores
2. Decode bounding box parameters to absolute coordinates
3. Apply confidence threshold (e.g., 0.3)
4. Apply Non-Maximum Suppression (NMS) with IoU threshold 0.5

**Key Design Decisions:**
- Dense prediction: one prediction per BEV grid cell
- Anchor-free design (direct coordinate regression)
- Multi-class detection (car, pedestrian, bicycle, etc.)

---

## Data Models

### BEVGridConfig
```python
@dataclass
class BEVGridConfig:
    x_min: float = -51.2  # meters
    x_max: float = 51.2
    y_min: float = -51.2
    y_max: float = 51.2
    z_min: float = -5.0
    z_max: float = 3.0
    resolution: float = 0.2  # meters per pixel
    
    @property
    def grid_size(self) -> Tuple[int, int]:
        H = int((self.y_max - self.y_min) / self.resolution)
        W = int((self.x_max - self.x_min) / self.resolution)
        return (H, W)
```

### Sample
```python
@dataclass
class Sample:
    # Identifiers
    sample_token: str
    scene_token: str
    timestamp: int
    
    # LiDAR data
    lidar_path: str
    lidar_points: Optional[np.ndarray] = None  # (N, 4)
    
    # Camera data
    camera_paths: Dict[str, str]  # 6 cameras
    camera_images: Optional[Dict[str, np.ndarray]] = None
    camera_intrinsics: Dict[str, np.ndarray]  # (3, 3)
    camera_extrinsics: Dict[str, np.ndarray]  # (4, 4)
    
    # Ego-motion
    ego_pose: np.ndarray  # (4, 4) SE(3) transform
    
    # Annotations
    annotations: List[Box3D]
```

### Box3D
```python
@dataclass
class Box3D:
    center: np.ndarray  # (3,) [x, y, z]
    size: np.ndarray    # (3,) [width, length, height]
    yaw: float          # rotation around z-axis
    label: str          # object category
    score: float = 1.0  # confidence score
    token: str = ""     # annotation token
```

### MemoryBank
```python
class MemoryBank:
    """Stores past BEV features for temporal aggregation."""
    def __init__(self, max_length: int = 5):
        self.max_length = max_length
        self.features: Deque[torch.Tensor] = deque(maxlen=max_length)
        self.transforms: Deque[torch.Tensor] = deque(maxlen=max_length-1)
    
    def push(self, feature: torch.Tensor, transform: Optional[torch.Tensor] = None):
        self.features.append(feature)
        if transform is not None:
            self.transforms.append(transform)
    
    def get_sequence(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return list(self.features), list(self.transforms)
    
    def clear(self):
        self.features.clear()
        self.transforms.clear()
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Point cloud to pillar conversion preserves all points within BEV bounds

*For any* point cloud with coordinates within the defined BEV range, converting to pillars should assign all points to valid pillar locations, and no points should be lost or assigned to out-of-bounds pillars.

**Validates: Requirements 1.1**

### Property 2: Pillar encoding produces consistent output dimensions

*For any* set of non-empty pillars, the PointNet-style encoder should produce features with shape (N_pillars, C_pillar) where C_pillar is the configured feature dimension, and all feature values should be finite.

**Validates: Requirements 1.2**

### Property 3: Scatter operation maintains pillar-to-grid correspondence

*For any* set of pillar features with coordinates, scattering to the BEV grid should place each pillar's features at the correct (x, y) grid location, and the output grid should have shape (B, C1, H, W).

**Validates: Requirements 1.3**

### Property 4: LiDAR BEV encoder output shape invariant

*For any* valid point cloud input, the complete LiDAR BEV encoder (pillarization + scatter + CNN) should output features with shape (B, C1, H, W) where C1, H, W match the configuration.

**Validates: Requirements 1.4, 1.5**

### Property 5: Camera backbone extracts features for all views

*For any* batch of multi-view images with shape (B, N_cam, 3, H_img, W_img), the image backbone should produce features for all N_cam views without errors, and all feature values should be finite.

**Validates: Requirements 2.1**

### Property 6: BEV cross-attention produces valid outputs

*For any* valid image features and BEV queries, the spatial cross-attention mechanism should produce output features without NaN or Inf values, and the output shape should match the BEV query shape.

**Validates: Requirements 2.2**

### Property 7: Camera BEV encoder output shape invariant

*For any* valid multi-view image input, the camera BEV encoder should output features with shape (B, C2, H, W) where C2, H, W match the configuration.

**Validates: Requirements 2.3**

### Property 8: Camera projection round-trip consistency

*For any* 3D point in BEV space, projecting to image space using camera parameters and then back-projecting to BEV should yield a point close to the original (within numerical tolerance).

**Validates: Requirements 2.4**

### Property 9: BEV grid alignment between modalities

*For any* LiDAR BEV features and camera BEV features, both should have the same spatial dimensions (H, W) and represent the same physical BEV region.

**Validates: Requirements 2.5, 3.3**

### Property 10: Spatial fusion output shape invariant

*For any* valid LiDAR BEV features F_lidar and camera BEV features F_cam with matching spatial dimensions, the fusion module should output F_fused with shape (B, C3, H, W) where C3 matches the configuration.

**Validates: Requirements 3.1, 3.2**

### Property 11: Temporal alignment preserves feature dimensions

*For any* sequence of BEV features and corresponding ego-motion transforms, aligning all features to the current frame should produce aligned features with the same shape as the original features.

**Validates: Requirements 4.1**

### Property 12: Temporal attention output validity

*For any* aligned sequence of BEV features, applying temporal self-attention should produce output features without NaN or Inf values, and the output shape should match the input feature shape.

**Validates: Requirements 4.2**

### Property 13: Temporal gating weights are bounded

*For any* temporal attention outputs, the computed gating weights should be in the range [0, 1] for all spatial locations.

**Validates: Requirements 4.3**

### Property 14: Residual connection preserves dimensions

*For any* current BEV features and temporally aggregated features, combining them via residual connection should produce output with the same shape as the current features.

**Validates: Requirements 4.4, 4.5**

### Property 15: Ego-motion warping preserves feature dimensions

*For any* BEV feature F_{t-1} and ego-motion transform, warping F_{t-1} should produce a warped feature with the same shape as F_{t-1}.

**Validates: Requirements 5.1**

### Property 16: Motion field has valid shape and range

*For any* pair of BEV features (current and warped previous), the estimated motion field should have shape (B, 2, H, W) and flow values should be bounded (e.g., within [-max_flow, max_flow]).

**Validates: Requirements 5.2**

### Property 17: Visibility mask values are valid

*For any* ego-motion transform, the computed visibility mask should have shape (B, 1, H, W) and all values should be in the range [0, 1].

**Validates: Requirements 5.3**

### Property 18: ConvGRU maintains hidden state dimensions

*For any* input features F_t and previous hidden state H_{t-1}, the ConvGRU should output new hidden state H_t with the same shape as H_{t-1}.

**Validates: Requirements 5.4, 5.5**

### Property 19: Detection head produces complete predictions

*For any* BEV features, the detection head should produce classification scores with shape (B, N_cls, H, W) and regression parameters with shape (B, 7, H, W).

**Validates: Requirements 6.1, 6.2, 6.3**

### Property 20: NMS reduces or maintains detection count

*For any* set of raw detections, applying NMS should produce a filtered set where the number of detections is less than or equal to the original count.

**Validates: Requirements 6.4**

### Property 21: Detection output format is valid

*For any* predictions after NMS, the output should be a list of Box3D objects where each box has confidence scores in [0, 1] and valid geometric parameters (positive sizes, yaw in [-π, π]).

**Validates: Requirements 6.5**

### Property 22: Dataset loading provides complete samples

*For any* valid sample token in the nuScenes dataset, loading should return exactly 1 LiDAR point cloud and 6 camera images, along with calibration matrices of correct shapes.

**Validates: Requirements 7.2, 7.3**

### Property 23: Ego-motion is available for consecutive frames

*For any* pair of consecutive samples in a scene, an ego-motion SE(3) transform should be computable from their ego-poses.

**Validates: Requirements 7.4**

### Property 24: Annotations are loaded for training samples

*For any* training sample, ground truth annotations should be available as a non-empty list of Box3D objects (unless the sample genuinely has no objects).

**Validates: Requirements 7.5**

### Property 25: Model parameters are properly initialized

*For any* newly created model, all parameters should be initialized (no None values), and parameter values should have reasonable magnitudes (e.g., not all zeros, not extremely large).

**Validates: Requirements 8.1**

### Property 26: Training losses are valid

*For any* training batch, computed losses (classification and regression) should be finite (not NaN or Inf) and non-negative.

**Validates: Requirements 8.2**

### Property 27: Gradients flow through the network

*For any* computed loss, backpropagation should produce non-zero gradients for at least some model parameters (indicating gradient flow).

**Validates: Requirements 8.3**

### Property 28: Checkpoint loading restores model state

*For any* valid checkpoint file, loading should successfully restore all model parameters, and the restored model should produce the same outputs as the model that created the checkpoint (given the same inputs).

**Validates: Requirements 9.1**

### Property 29: Evaluation metrics are in valid ranges

*For any* set of predictions and ground truth annotations, computed metrics (AP, NDS) should have values in their valid ranges (e.g., AP in [0, 1], NDS in [0, 1]).

**Validates: Requirements 9.3**

### Property 30: Configuration parameters are applied correctly

*For any* valid configuration file specifying BEV dimensions, feature dimensions, and temporal sequence length, the system should initialize all modules using those configured values.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**

---

## Error Handling

### Input Validation

**Point Cloud Validation:**
- Check for NaN or Inf values in point coordinates
- Verify points are within reasonable range (e.g., within 100m of ego vehicle)
- Handle empty point clouds gracefully (return zero features)

**Image Validation:**
- Check image dimensions match expected input size
- Verify pixel values are in valid range [0, 255] or [0, 1]
- Handle missing camera views (mask out in attention)

**Calibration Validation:**
- Verify intrinsic matrices are 3×3 and extrinsic matrices are 4×4
- Check for degenerate calibration (e.g., zero focal length)
- Validate that projection matrices are invertible

### Runtime Error Handling

**Out-of-Memory:**
- Implement gradient checkpointing for large models
- Support mixed precision training (FP16)
- Provide clear error messages with memory usage statistics

**Numerical Instability:**
- Add epsilon values to prevent division by zero
- Clip gradients to prevent explosion
- Use stable softmax implementations

**Missing Data:**
- Handle missing temporal history (use only current frame)
- Support variable sequence lengths in temporal module
- Gracefully degrade when sensors fail

### Training Error Handling

**Loss Divergence:**
- Monitor for NaN/Inf losses and terminate training
- Implement loss scaling for numerical stability
- Log gradient norms for debugging

**Checkpoint Corruption:**
- Validate checkpoint integrity before loading
- Keep multiple checkpoint backups
- Support resuming from last valid checkpoint

---

## Testing Strategy

### Unit Testing

**Component-Level Tests:**
- Test each module independently with synthetic inputs
- Verify output shapes and value ranges
- Test edge cases (empty inputs, single-point clouds, etc.)

**Key Unit Tests:**
1. `test_pillar_encoding`: Verify pillar features have correct shape
2. `test_bev_scatter`: Verify scatter operation places features correctly
3. `test_cross_attention`: Verify attention produces valid outputs
4. `test_ego_motion_warping`: Verify warping preserves feature dimensions
5. `test_convgru_update`: Verify hidden state updates correctly
6. `test_detection_head`: Verify prediction shapes are correct
7. `test_nms`: Verify NMS reduces overlapping detections

### Property-Based Testing

The system will use **Hypothesis** (Python property-based testing library) to verify the correctness properties defined above.

**Configuration:**
- Minimum 100 iterations per property test
- Use random seeds for reproducibility
- Generate diverse test cases (various batch sizes, feature dimensions, etc.)

**Test Generators:**

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

# Strategy for generating valid point clouds
@st.composite
def point_clouds(draw, min_points=10, max_points=10000):
    n_points = draw(st.integers(min_value=min_points, max_value=max_points))
    x = draw(npst.arrays(dtype=np.float32, shape=(n_points,), 
                         elements=st.floats(min_value=-50, max_value=50)))
    y = draw(npst.arrays(dtype=np.float32, shape=(n_points,), 
                         elements=st.floats(min_value=-50, max_value=50)))
    z = draw(npst.arrays(dtype=np.float32, shape=(n_points,), 
                         elements=st.floats(min_value=-5, max_value=3)))
    r = draw(npst.arrays(dtype=np.float32, shape=(n_points,), 
                         elements=st.floats(min_value=0, max_value=1)))
    return np.stack([x, y, z, r], axis=1)

# Strategy for generating BEV features
@st.composite
def bev_features(draw, batch_size=2, channels=64, height=200, width=200):
    return draw(npst.arrays(dtype=np.float32, 
                           shape=(batch_size, channels, height, width),
                           elements=st.floats(min_value=-10, max_value=10)))

# Strategy for generating ego-motion transforms
@st.composite
def ego_motion_transforms(draw):
    # Generate random SE(3) transform
    translation = draw(npst.arrays(dtype=np.float32, shape=(3,),
                                   elements=st.floats(min_value=-5, max_value=5)))
    # Generate random rotation (simplified: only yaw rotation)
    yaw = draw(st.floats(min_value=-np.pi, max_value=np.pi))
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform
```

**Property Test Examples:**

```python
# Property 1: Point cloud to pillar conversion
@given(points=point_clouds())
def test_property_1_pillar_conversion(points):
    """Feature: bev-fusion-system, Property 1: Point cloud to pillar conversion preserves all points within BEV bounds"""
    config = BEVGridConfig()
    encoder = LiDARBEVEncoder(config)
    
    # Filter points within BEV bounds
    mask = (points[:, 0] >= config.x_min) & (points[:, 0] <= config.x_max) & \
           (points[:, 1] >= config.y_min) & (points[:, 1] <= config.y_max)
    points_in_bounds = points[mask]
    
    # Convert to pillars
    pillars, coords = encoder.pillarize(points)
    
    # Verify all pillars are within grid bounds
    H, W = config.grid_size
    assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] < H)
    assert np.all(coords[:, 1] >= 0) and np.all(coords[:, 1] < W)
    
    # Verify no points are lost (within tolerance for pillar aggregation)
    assert len(pillars) > 0 if len(points_in_bounds) > 0 else len(pillars) == 0

# Property 4: LiDAR BEV encoder output shape
@given(points=point_clouds(), batch_size=st.integers(min_value=1, max_value=4))
def test_property_4_lidar_output_shape(points, batch_size):
    """Feature: bev-fusion-system, Property 4: LiDAR BEV encoder output shape invariant"""
    config = BEVGridConfig()
    encoder = LiDARBEVEncoder(config)
    
    # Create batch
    points_batch = [points for _ in range(batch_size)]
    
    # Forward pass
    output = encoder(points_batch)
    
    # Verify output shape
    H, W = config.grid_size
    assert output.shape == (batch_size, config.C1, H, W)
    assert torch.isfinite(output).all()

# Property 8: Camera projection round-trip
@given(points_3d=npst.arrays(dtype=np.float32, shape=(10, 3),
                              elements=st.floats(min_value=-50, max_value=50)))
def test_property_8_projection_round_trip(points_3d):
    """Feature: bev-fusion-system, Property 8: Camera projection round-trip consistency"""
    # Create synthetic camera parameters
    intrinsic = np.array([[1000, 0, 800], [0, 1000, 600], [0, 0, 1]], dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    
    # Project to image
    points_2d = project_3d_to_2d(points_3d, intrinsic, extrinsic)
    
    # Back-project to 3D (assuming known depth)
    points_3d_reconstructed = backproject_2d_to_3d(points_2d, points_3d[:, 2], 
                                                    intrinsic, extrinsic)
    
    # Verify round-trip consistency
    np.testing.assert_allclose(points_3d, points_3d_reconstructed, rtol=1e-4, atol=1e-4)

# Property 15: Ego-motion warping preserves dimensions
@given(features=bev_features(), ego_motion=ego_motion_transforms())
def test_property_15_warping_preserves_dimensions(features, ego_motion):
    """Feature: bev-fusion-system, Property 15: Ego-motion warping preserves feature dimensions"""
    warper = EgoMotionWarper()
    
    # Warp features
    warped_features = warper(torch.from_numpy(features), torch.from_numpy(ego_motion))
    
    # Verify shape preservation
    assert warped_features.shape == features.shape
    assert torch.isfinite(warped_features).all()

# Property 20: NMS reduces detection count
@given(boxes=st.lists(st.tuples(
    st.floats(min_value=-50, max_value=50),  # x
    st.floats(min_value=-50, max_value=50),  # y
    st.floats(min_value=0, max_value=10),    # w
    st.floats(min_value=0, max_value=10),    # l
    st.floats(min_value=0, max_value=1)      # score
), min_size=0, max_size=100))
def test_property_20_nms_reduces_count(boxes):
    """Feature: bev-fusion-system, Property 20: NMS reduces or maintains detection count"""
    if len(boxes) == 0:
        return
    
    # Convert to Box3D format
    box_objects = [Box3D(center=np.array([x, y, 0]), 
                         size=np.array([w, l, 2]), 
                         yaw=0, label='car', score=score)
                   for x, y, w, l, score in boxes]
    
    # Apply NMS
    filtered_boxes = apply_nms(box_objects, iou_threshold=0.5)
    
    # Verify count reduction
    assert len(filtered_boxes) <= len(box_objects)
```

### Integration Testing

**End-to-End Tests:**
1. Test complete pipeline from raw data to predictions
2. Verify training loop completes without errors
3. Test checkpoint saving and loading
4. Verify evaluation metrics are computed correctly

**Data Flow Tests:**
- Test data loading and batching
- Verify data augmentation preserves annotations
- Test temporal sequence construction

### Performance Testing

**Throughput Benchmarks:**
- Measure inference time per sample
- Measure training time per epoch
- Profile memory usage

**Targets:**
- Inference: < 100ms per frame (10 FPS)
- Training: Complete 1 epoch on nuScenes-mini in < 30 minutes
- Memory: Fit batch size 2 in 16GB GPU memory

---

## Implementation Notes

### Dependencies

**Core Libraries:**
- PyTorch >= 2.0
- torchvision >= 0.15
- numpy >= 1.24
- scipy >= 1.10

**Dataset and Evaluation:**
- nuscenes-devkit >= 1.1.9
- pyquaternion >= 0.9.9

**Utilities:**
- pyyaml >= 6.0 (configuration)
- tqdm >= 4.65 (progress bars)
- tensorboard >= 2.12 (logging)

**Testing:**
- pytest >= 7.3
- hypothesis >= 6.75 (property-based testing)

### Development Environment

**Python Environment:**
- Use `uv` for package management
- Python 3.10 or 3.11
- CUDA 11.8 or 12.0 for GPU support

**Installation:**
```bash
# Install PyTorch with CUDA support
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# Install other dependencies
uv pip install nuscenes-devkit pyquaternion pyyaml tqdm tensorboard pytest hypothesis
```

### Project Structure

```
bev-fusion-system/
├── modules/
│   ├── __init__.py
│   ├── lidar_encoder.py      # PointPillars implementation
│   ├── camera_encoder.py     # BEVFormer implementation
│   ├── fusion.py              # Spatial fusion module
│   ├── temporal.py            # Temporal aggregation modules
│   │   ├── attention.py       # Transformer-based temporal attention
│   │   ├── mc_convrnn.py      # Motion-compensated ConvRNN
│   ├── detection_head.py      # 3D detection head
│   ├── dataset.py             # nuScenes dataset loader
│   ├── utils.py               # Utility functions
│   └── losses.py              # Loss functions
├── configs/
│   ├── base_config.py         # Base configuration
│   ├── train_config.yaml      # Training configuration
│   └── eval_config.yaml       # Evaluation configuration
├── tests/
│   ├── unit/
│   │   ├── test_lidar_encoder.py
│   │   ├── test_camera_encoder.py
│   │   ├── test_fusion.py
│   │   ├── test_temporal.py
│   │   └── test_detection_head.py
│   ├── property/
│   │   ├── test_properties.py  # Property-based tests
│   │   └── strategies.py       # Hypothesis strategies
│   └── integration/
│       ├── test_pipeline.py
│       └── test_training.py
├── scripts/
│   ├── prepare_data.py        # Data preparation script
│   ├── visualize.py           # Visualization utilities
│   └── benchmark.py           # Performance benchmarking
├── train.py                   # Training script
├── eval.py                    # Evaluation script
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

### Configuration Management

Use YAML files for configuration with Python dataclasses for type safety:

```python
from dataclasses import dataclass
from typing import Tuple
import yaml

@dataclass
class ModelConfig:
    # BEV grid
    bev_h: int = 200
    bev_w: int = 200
    bev_resolution: float = 0.2
    
    # Feature dimensions
    lidar_channels: int = 64
    camera_channels: int = 256
    fused_channels: int = 256
    
    # Temporal
    temporal_length: int = 5
    use_temporal_attention: bool = True
    use_mc_convrnn: bool = True
    
    # Detection
    num_classes: int = 10
    nms_threshold: float = 0.5
    score_threshold: float = 0.3

@dataclass
class TrainingConfig:
    batch_size: int = 2
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 35.0
    
    # Checkpointing
    checkpoint_interval: int = 1
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_interval: int = 10
    tensorboard_dir: str = "runs"

def load_config(config_path: str) -> Tuple[ModelConfig, TrainingConfig]:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    model_config = ModelConfig(**config_dict['model'])
    train_config = TrainingConfig(**config_dict['training'])
    return model_config, train_config
```

### Logging and Monitoring

**TensorBoard Integration:**
- Log training/validation losses
- Log learning rate schedule
- Log gradient norms
- Visualize predictions on validation samples

**Console Logging:**
- Use `tqdm` for progress bars
- Print epoch summaries
- Report evaluation metrics

### Checkpointing Strategy

**Save:**
- Model state dict
- Optimizer state dict
- Epoch number
- Best validation metric
- Configuration used

**Load:**
- Support resuming from checkpoint
- Support loading only model weights (for evaluation)
- Validate checkpoint compatibility with current config

---

This design provides a comprehensive blueprint for implementing the BEV fusion system with clear interfaces, correctness properties, and testing strategies.
