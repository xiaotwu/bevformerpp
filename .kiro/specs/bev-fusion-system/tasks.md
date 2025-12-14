# Implementation Plan

- [x] 1. Set up project structure and environment






  - Create directory structure for modules, configs, tests, and scripts
  - Set up Python environment with uv
  - Install core dependencies (PyTorch, nuScenes devkit, testing frameworks)
  - Create base configuration files
  - _Requirements: 10.5_

- [x] 2. Implement data preparation and dataset loading





  - [x] 2.1 Create dataset extraction script


    - Extract v1.0-mini.tgz to data directory
    - Verify dataset structure and files
    - _Requirements: 7.1_
  
  - [x] 2.2 Implement nuScenes dataset loader


    - Create Sample and Box3D dataclasses
    - Implement dataset class to load LiDAR, images, calibration, and annotations
    - Handle synchronized multi-sensor data loading
    - Implement ego-motion computation between frames
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [x] 2.3 Implement data preprocessing utilities


    - Point cloud filtering and normalization
    - Image resizing and normalization
    - BEV coordinate transformations
    - _Requirements: 1.1, 2.1_
  
  - [x] 2.4 Write unit tests for dataset loading



    - Test sample loading completeness
    - Test calibration matrix shapes
    - Test ego-motion computation
    - _Requirements: 7.2, 7.3, 7.4_

- [x] 3. Implement LiDAR BEV encoder (PointPillars)




  - [x] 3.1 Implement pillarization module


    - Create BEVGridConfig dataclass
    - Implement point-to-pillar assignment
    - Handle pillar coordinate computation
    - _Requirements: 1.1_
  
  - [x] 3.2 Implement PillarFeatureNet

    - PointNet-style per-pillar encoding
    - Feature aggregation within pillars
    - _Requirements: 1.2_
  
  - [x] 3.3 Implement PointPillarsScatter

    - Scatter pillar features to dense BEV grid
    - Handle empty pillars with zero-padding
    - _Requirements: 1.3_
  
  - [x] 3.4 Implement 2D CNN backbone

    - Residual blocks for BEV feature refinement
    - Progressive feature extraction
    - _Requirements: 1.4_
  
  - [x] 3.5 Integrate LiDAR encoder components

    - Create LiDARBEVEncoder module combining all sub-components
    - Verify output shape (B, C1, H, W)
    - _Requirements: 1.5_
  
  - [x] 3.6 Write property test for pillar conversion


    - **Property 1: Point cloud to pillar conversion preserves all points within BEV bounds**
    - **Validates: Requirements 1.1**
  
  - [x] 3.7 Write property test for LiDAR output shape


    - **Property 4: LiDAR BEV encoder output shape invariant**
    - **Validates: Requirements 1.4, 1.5**

- [x] 4. Implement camera BEV encoder (BEVFormer)





  - [x] 4.1 Implement image backbone


    - Load pretrained ResNet50
    - Extract multi-scale features
    - _Requirements: 2.1_
  
  - [x] 4.2 Implement spatial cross-attention


    - Initialize learnable BEV queries with positional encoding
    - Implement deformable attention mechanism
    - Project image features to BEV using camera geometry
    - _Requirements: 2.2, 2.4_
  
  - [x] 4.3 Implement BEVFormerEncoder


    - Stack multiple cross-attention layers
    - Integrate with image backbone
    - Output camera BEV features
    - _Requirements: 2.3_
  
  - [x] 4.4 Implement camera projection utilities


    - 3D to 2D projection using intrinsics/extrinsics
    - 2D to 3D back-projection
    - _Requirements: 2.4_
  
  - [x] 4.5 Write property test for camera output shape


    - **Property 7: Camera BEV encoder output shape invariant**
    - **Validates: Requirements 2.3**
  
  - [x] 4.6 Write property test for projection round-trip


    - **Property 8: Camera projection round-trip consistency**
    - **Validates: Requirements 2.4**

- [x] 5. Implement spatial fusion module





  - [x] 5.1 Implement cross-attention fusion


    - Multi-head attention between LiDAR and camera BEV features
    - Layer normalization and residual connections
    - _Requirements: 3.1_
  
  - [x] 5.2 Verify BEV grid alignment


    - Ensure LiDAR and camera BEV have matching dimensions
    - Validate spatial correspondence
    - _Requirements: 2.5, 3.3_
  
  - [x] 5.3 Create SpatialFusionModule

    - Integrate fusion mechanism
    - Output unified BEV features
    - _Requirements: 3.2_
  
  - [x] 5.4 Write property test for fusion output shape



    - **Property 10: Spatial fusion output shape invariant**
    - **Validates: Requirements 3.1, 3.2**
  
  - [x] 5.5 Write property test for BEV alignment

    - **Property 9: BEV grid alignment between modalities**
    - **Validates: Requirements 2.5, 3.3**

- [x] 6. Implement temporal aggregation - Transformer-based attention




  - [x] 6.1 Implement MemoryBank for feature storage


    - FIFO queue for past BEV features
    - Store ego-motion transforms
    - _Requirements: 4.1_
  
  - [x] 6.2 Implement ego-motion alignment


    - Warp past features to current frame
    - Grid sampling with bilinear interpolation
    - _Requirements: 4.1_
  
  - [x] 6.3 Implement temporal self-attention


    - Multi-head attention across time sequence
    - Deformable attention for efficiency
    - _Requirements: 4.2_
  
  - [x] 6.4 Implement temporal gating


    - Compute confidence weights for each frame
    - Adaptive weighting based on feature similarity
    - _Requirements: 4.3_
  
  - [x] 6.5 Implement residual update


    - Combine current and temporally aggregated features
    - _Requirements: 4.4_
  
  - [x] 6.6 Create TemporalSelfAttention module


    - Integrate all temporal attention components
    - _Requirements: 4.5_
  
  - [x] 6.7 Write property test for temporal alignment



    - **Property 11: Temporal alignment preserves feature dimensions**
    - **Validates: Requirements 4.1**
  
  - [x] 6.8 Write property test for gating weights

    - **Property 13: Temporal gating weights are bounded**
    - **Validates: Requirements 4.3**

- [x] 7. Implement temporal aggregation - Motion-Compensated ConvRNN


  - [x] 7.1 Implement ego-motion warping


    - Generate sampling grid from SE(3) transform
    - Warp features using grid sampling
    - _Requirements: 5.1_
  
  - [x] 7.2 Implement motion field estimator


    - Lightweight CNN for residual motion estimation
    - Output 2D flow field
    - _Requirements: 5.2_
  
  - [x] 7.3 Implement visibility gating

    - Compute visibility mask from ego-motion
    - Apply mask to warped features
    - _Requirements: 5.3_
  
  - [x] 7.4 Implement ConvGRU cell

    - Update and reset gates
    - Hidden state update
    - _Requirements: 5.4_
  
  - [x] 7.5 Create MCConvRNN module

    - Integrate warping, motion field, visibility, and ConvGRU
    - Output motion-compensated features
    - _Requirements: 5.5_
  
  - [x] 7.6 Write property test for warping

    - **Property 15: Ego-motion warping preserves feature dimensions**
    - **Validates: Requirements 5.1**
  
  - [x] 7.7 Write property test for motion field

    - **Property 16: Motion field has valid shape and range**
    - **Validates: Requirements 5.2**
  
  - [x] 7.8 Write property test for ConvGRU

    - **Property 18: ConvGRU maintains hidden state dimensions**
    - **Validates: Requirements 5.4, 5.5**

- [x] 8. Implement detection head


  - [x] 8.1 Implement shared convolutional layers


    - Feature refinement before prediction
    - _Requirements: 6.1_
  
  - [x] 8.2 Implement classification head

    - Dense prediction of class scores
    - Output shape (B, N_cls, H, W)
    - _Requirements: 6.2_
  
  - [x] 8.3 Implement regression head

    - Dense prediction of bounding box parameters
    - Output shape (B, 7, H, W) for (x, y, z, w, l, h, yaw)
    - _Requirements: 6.3_
  
  - [x] 8.4 Implement NMS post-processing


    - IoU computation for 3D boxes
    - Non-maximum suppression
    - _Requirements: 6.4_
  
  - [x] 8.5 Implement detection decoding

    - Convert predictions to Box3D objects
    - Apply confidence thresholding
    - _Requirements: 6.5_
  
  - [x] 8.6 Write property test for detection head output

    - **Property 19: Detection head produces complete predictions**
    - **Validates: Requirements 6.1, 6.2, 6.3**
  
  - [x] 8.7 Write property test for NMS

    - **Property 20: NMS reduces or maintains detection count**
    - **Validates: Requirements 6.4**

- [x] 9. Implement loss functions


  - [x] 9.1 Implement classification loss

    - Focal loss for class imbalance
    - Handle positive/negative samples
    - _Requirements: 8.2_
  
  - [x] 9.2 Implement regression loss

    - Smooth L1 loss for bounding box parameters
    - Separate losses for center, size, and rotation
    - _Requirements: 8.2_
  
  - [x] 9.3 Implement total loss computation

    - Weighted combination of classification and regression losses
    - _Requirements: 8.2_
  
  - [x] 9.4 Write property test for loss validity

    - **Property 26: Training losses are valid**
    - **Validates: Requirements 8.2**

- [x] 10. Integrate full pipeline



  - [x] 10.1 Create BEVFusionModel top-level module


    - Integrate all components (LiDAR, camera, fusion, temporal, detection)
    - Implement forward pass
    - _Requirements: 1.1-6.5_
  
  - [x] 10.2 Implement configuration loading


    - Load model and training configs from YAML
    - Initialize modules with configured parameters
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [x] 10.3 Verify end-to-end forward pass

    - Test with synthetic data
    - Verify output shapes and value ranges
    - _Requirements: 1.1-6.5_
  
  - [x] 10.4 Write property test for configuration

    - **Property 30: Configuration parameters are applied correctly**
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**

- [x] 11. Checkpoint - Ensure all tests pass






  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement training loop




  - [x] 12.1 Implement model initialization
    - Random weight initialization
    - Load pretrained weights for image backbone
    - _Requirements: 8.1_
  

  - [x] 12.2 Implement training step
    - Forward pass
    - Loss computation
    - Backward pass and optimizer step
    - _Requirements: 8.2, 8.3_

  
  - [x] 12.3 Implement gradient clipping
    - Clip gradients to prevent explosion
    - _Requirements: 8.3_
  

  - [x] 12.4 Implement logging
    - TensorBoard integration
    - Console logging with tqdm
    - _Requirements: 8.4_

  
  - [x] 12.5 Implement checkpointing
    - Save model, optimizer, and training state
    - Load checkpoint for resuming
    - _Requirements: 8.5_
  

  - [x] 12.6 Create training script (train.py)
    - Main training loop
    - Epoch iteration
    - Validation after each epoch
    - _Requirements: 8.1-8.5_

  
  - [x] 12.7 Write property test for gradient flow
    - **Property 27: Gradients flow through the network**
    - **Validates: Requirements 8.3**

- [x] 13. Implement evaluation
  - [x] 13.1 Implement inference mode
    - Disable gradient computation
    - Batch processing of validation samples
    - _Requirements: 9.2_
  
  - [x] 13.2 Implement metric computation
    - Average Precision (AP) calculation
    - nuScenes Detection Score (NDS) calculation
    - _Requirements: 9.3_
  
  - [x] 13.3 Implement result formatting
    - Print metrics in readable format
    - Save results to file
    - _Requirements: 9.4_
  
  - [x] 13.4 Implement visualization
    - Overlay predictions on images
    - Visualize BEV detections
    - _Requirements: 9.5_
  
  - [x] 13.5 Create evaluation script (eval.py)
    - Load checkpoint
    - Run inference on validation set
    - Compute and report metrics
    - _Requirements: 9.1-9.5_
  
  - [x] 13.6 Write property test for checkpoint loading
    - **Property 28: Checkpoint loading restores model state**
    - **Validates: Requirements 9.1**
  
  - [x] 13.7 Write property test for metrics
    - **Property 29: Evaluation metrics are in valid ranges**
    - **Validates: Requirements 9.3**

- [x] 14. Create main entry point and utilities
  - [x] 14.1 Create main.py
    - Command-line interface for training and evaluation
    - Argument parsing
    - _Requirements: 8.1-9.5_
  
  - [x] 14.2 Create visualization utilities
    - BEV visualization
    - 3D box rendering
    - Multi-view image display
    - _Requirements: 9.5_
  
  - [x] 14.3 Create benchmarking script
    - Measure inference time
    - Profile memory usage
    - _Requirements: 8.1-9.5_
  
  - [x] 14.4 Update README with usage instructions
    - Installation steps
    - Training and evaluation commands
    - Configuration guide
    - _Requirements: 10.5_

- [x] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Run baseline experiments
  - [x] 16.1 Train on nuScenes v1.0-mini
    - Run training for 20 epochs
    - Monitor training metrics
    - _Requirements: 8.1-8.5_
  
  - [x] 16.2 Evaluate on validation set
    - Compute AP and NDS metrics
    - Generate visualizations
    - _Requirements: 9.1-9.5_
  
  - [x] 16.3 Document results
    - Record metrics and training time
    - Save sample visualizations
    - _Requirements: 9.4_
