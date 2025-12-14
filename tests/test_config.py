"""Tests for configuration loading and management."""

import pytest
from pathlib import Path
import yaml

from configs import load_config, merge_configs, Config


class TestConfigLoader:
    """Test suite for configuration loader."""
    
    def test_load_base_config(self, config_path):
        """Test loading base configuration file."""
        config = load_config(config_path)
        
        # Check that configuration is loaded
        assert config is not None
        assert isinstance(config, Config)
        
        # Check key sections exist
        assert hasattr(config, "bev_grid")
        assert hasattr(config, "model")
        assert hasattr(config, "dataset")
        assert hasattr(config, "training")
        assert hasattr(config, "evaluation")
    
    def test_config_dot_notation(self, config_path):
        """Test that configuration supports dot notation access."""
        config = load_config(config_path)
        
        # Test nested access
        assert config.bev_grid.x_min == -51.2
        assert config.bev_grid.x_max == 51.2
        assert config.model.lidar.num_features == 64
        assert config.model.camera.num_features == 256
        assert config.training.batch_size == 2
    
    def test_config_dict_access(self, config_path):
        """Test that configuration supports dictionary-style access."""
        config = load_config(config_path)
        
        # Test dictionary access
        assert config["bev_grid"]["x_min"] == -51.2
        assert config["model"]["lidar"]["num_features"] == 64
    
    def test_config_to_dict(self, config_path):
        """Test converting configuration back to dictionary."""
        config = load_config(config_path)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "bev_grid" in config_dict
        assert "model" in config_dict
        assert config_dict["bev_grid"]["x_min"] == -51.2
    
    def test_merge_configs(self):
        """Test merging two configuration dictionaries."""
        base = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": 4
        }
        
        override = {
            "b": {"c": 10},
            "f": 5
        }
        
        merged = merge_configs(base, override)
        
        # Check that values are correctly merged
        assert merged["a"] == 1  # Unchanged
        assert merged["b"]["c"] == 10  # Overridden
        assert merged["b"]["d"] == 3  # Preserved from base
        assert merged["e"] == 4  # Unchanged
        assert merged["f"] == 5  # Added from override
    
    def test_config_with_base_reference(self, tmp_path):
        """Test loading configuration that references a base configuration."""
        # Create base config
        base_config = {
            "param1": 10,
            "param2": {"nested": 20}
        }
        base_path = tmp_path / "base.yaml"
        with open(base_path, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create override config
        override_config = {
            "base": "base.yaml",
            "param1": 15,
            "param3": 30
        }
        override_path = tmp_path / "override.yaml"
        with open(override_path, 'w') as f:
            yaml.dump(override_config, f)
        
        # Load configuration
        config = load_config(override_path)
        
        # Check merged values
        assert config.param1 == 15  # Overridden
        assert config.param2.nested == 20  # From base
        assert config.param3 == 30  # New parameter
    
    def test_config_get_method(self, config_path):
        """Test Config.get() method with defaults."""
        config = load_config(config_path)
        
        # Test existing key
        assert config.get("bev_grid") is not None
        
        # Test non-existing key with default
        assert config.get("nonexistent", "default_value") == "default_value"
    
    def test_config_contains(self, config_path):
        """Test checking if key exists in configuration."""
        config = load_config(config_path)
        
        assert "bev_grid" in config
        assert "model" in config
        assert "nonexistent_key" not in config


class TestBEVGridConfig:
    """Test suite for BEV grid configuration parameters."""
    
    def test_bev_grid_parameters(self, config_path):
        """Test that BEV grid parameters are correctly loaded."""
        config = load_config(config_path)
        
        assert config.bev_grid.x_min == -51.2
        assert config.bev_grid.x_max == 51.2
        assert config.bev_grid.y_min == -51.2
        assert config.bev_grid.y_max == 51.2
        assert config.bev_grid.z_min == -5.0
        assert config.bev_grid.z_max == 3.0
        assert config.bev_grid.resolution == 0.2


class TestModelConfig:
    """Test suite for model configuration parameters."""
    
    def test_lidar_config(self, config_path):
        """Test LiDAR encoder configuration."""
        config = load_config(config_path)
        
        assert config.model.lidar.num_features == 64
        assert config.model.lidar.pillar_channels == 64
        assert config.model.lidar.max_points_per_pillar == 32
    
    def test_camera_config(self, config_path):
        """Test camera encoder configuration."""
        config = load_config(config_path)
        
        assert config.model.camera.backbone == "resnet50"
        assert config.model.camera.pretrained is True
        assert config.model.camera.num_features == 256
        assert config.model.camera.num_attention_heads == 8
    
    def test_fusion_config(self, config_path):
        """Test fusion module configuration."""
        config = load_config(config_path)
        
        assert config.model.fusion.num_features == 256
        assert config.model.fusion.fusion_type in ["cross_attention", "concat_conv"]
    
    def test_temporal_config(self, config_path):
        """Test temporal module configuration."""
        config = load_config(config_path)
        
        assert config.model.temporal.sequence_length == 5
        assert config.model.temporal.use_transformer is True
        assert config.model.temporal.use_convgru is True
    
    def test_detection_config(self, config_path):
        """Test detection head configuration."""
        config = load_config(config_path)
        
        assert config.model.detection.num_classes == 10
        assert config.model.detection.bbox_params == 7
        assert config.model.detection.nms_iou_threshold == 0.5
