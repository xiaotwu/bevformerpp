# BEV Fusion System Tests

This directory contains the test suite for the BEV Fusion System.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures and configuration
├── test_config.py           # Configuration loading tests
├── test_dataset.py          # Dataset loading tests (to be implemented)
├── test_lidar_encoder.py    # LiDAR encoder tests (to be implemented)
├── test_camera_encoder.py   # Camera encoder tests (to be implemented)
├── test_fusion.py           # Spatial fusion tests (to be implemented)
├── test_temporal.py         # Temporal aggregation tests (to be implemented)
├── test_detection.py        # Detection head tests (to be implemented)
└── property_tests/          # Property-based tests using Hypothesis
    ├── test_properties_lidar.py
    ├── test_properties_camera.py
    ├── test_properties_fusion.py
    ├── test_properties_temporal.py
    └── test_properties_detection.py
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_config.py
```

### Run tests with coverage
```bash
pytest tests/ --cov=modules --cov-report=html
```

### Run only unit tests
```bash
pytest tests/ -m unit
```

### Run only property-based tests
```bash
pytest tests/ -m property
```

### Run tests in verbose mode
```bash
pytest tests/ -v
```

## Test Markers

- `unit`: Unit tests for individual components
- `property`: Property-based tests using Hypothesis
- `integration`: Integration tests for multiple components
- `slow`: Tests that take a long time to run

## Writing Tests

### Unit Tests

Unit tests should focus on testing individual components in isolation:

```python
import pytest
from modules.lidar_encoder import LiDARBEVEncoder

@pytest.mark.unit
def test_lidar_encoder_output_shape(sample_point_cloud):
    encoder = LiDARBEVEncoder(config)
    output = encoder(sample_point_cloud)
    assert output.shape == (batch_size, 64, 200, 200)
```

### Property-Based Tests

Property-based tests use Hypothesis to generate random test cases:

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@pytest.mark.property
@given(points=point_clouds())
def test_property_pillar_conversion(points):
    """Feature: bev-fusion-system, Property 1: Point cloud to pillar conversion preserves all points within BEV bounds"""
    encoder = LiDARBEVEncoder(config)
    # Test implementation
```

## Fixtures

Common fixtures are defined in `conftest.py`:

- `device`: PyTorch device (cuda/cpu)
- `batch_size`: Default batch size
- `bev_grid_size`: BEV grid dimensions
- `sample_point_cloud`: Sample LiDAR point cloud
- `sample_images`: Sample multi-view images
- `sample_bev_features`: Sample BEV features
- `sample_ego_motion`: Sample ego-motion transforms
- `config_path`: Path to test configuration
- `temp_checkpoint_dir`: Temporary directory for checkpoints
