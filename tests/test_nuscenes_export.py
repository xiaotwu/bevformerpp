"""
Tests for nuScenes export schema validation (requirement 2.2).

These tests verify that:
1. Export produces correct JSON schema with 'meta' and 'results' keys
2. Meta contains required boolean flags
3. Results is a dict mapping sample_token to list of detections
4. Each detection has all required fields with correct types
"""

import pytest
import json
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip if export module not available
pytest.importorskip("scripts.eval.export_nuscenes_predictions")

from scripts.eval.export_nuscenes_predictions import (
    create_nuscenes_submission,
    validate_submission_schema,
    normalize_class_name,
    yaw_to_quaternion,
    box3d_to_nuscenes_detection,
    NUSCENES_DETECTION_CLASSES
)
from modules.data_structures import Box3D


class TestNuScenesSubmissionSchema:
    """Tests for nuScenes submission JSON schema."""

    def test_create_submission_has_meta_and_results(self):
        """Test that submission has required top-level keys."""
        results = {
            "sample_token_1": [],
            "sample_token_2": []
        }
        submission = create_nuscenes_submission(results)

        assert "meta" in submission
        assert "results" in submission

    def test_meta_has_required_keys(self):
        """Test that meta contains all required modality flags."""
        submission = create_nuscenes_submission({})

        required_keys = ["use_camera", "use_lidar", "use_radar", "use_map", "use_external"]
        for key in required_keys:
            assert key in submission["meta"], f"Missing meta key: {key}"
            assert isinstance(submission["meta"][key], bool), f"meta[{key}] must be bool"

    def test_results_is_dict(self):
        """Test that results is a dict."""
        submission = create_nuscenes_submission({"token1": []})
        assert isinstance(submission["results"], dict)

    def test_valid_submission_passes_validation(self):
        """Test that a properly formatted submission passes validation."""
        detection = {
            "sample_token": "abc123",
            "translation": [1.0, 2.0, 3.0],
            "size": [1.5, 2.0, 1.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0],
            "detection_name": "car",
            "detection_score": 0.9,
            "attribute_name": ""
        }
        results = {"abc123": [detection]}
        submission = create_nuscenes_submission(results)

        # Should not raise
        assert validate_submission_schema(submission) is True

    def test_empty_results_valid(self):
        """Test that empty results (no detections) is valid."""
        submission = create_nuscenes_submission({})
        assert validate_submission_schema(submission) is True

    def test_sample_with_no_detections_valid(self):
        """Test that samples with empty detection lists are valid."""
        submission = create_nuscenes_submission({"token1": [], "token2": []})
        assert validate_submission_schema(submission) is True


class TestSchemaValidationErrors:
    """Tests for schema validation error handling."""

    def test_missing_meta_raises(self):
        """Test that missing meta key raises ValueError."""
        submission = {"results": {}}
        with pytest.raises(ValueError, match="missing 'meta'"):
            validate_submission_schema(submission)

    def test_missing_results_raises(self):
        """Test that missing results key raises ValueError."""
        submission = {"meta": {
            "use_camera": True, "use_lidar": True,
            "use_radar": False, "use_map": False, "use_external": False
        }}
        with pytest.raises(ValueError, match="missing 'results'"):
            validate_submission_schema(submission)

    def test_non_bool_meta_raises(self):
        """Test that non-boolean meta values raise ValueError."""
        submission = {
            "meta": {
                "use_camera": "true",  # String instead of bool
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False
            },
            "results": {}
        }
        with pytest.raises(ValueError, match="must be bool"):
            validate_submission_schema(submission)

    def test_invalid_detection_name_raises(self):
        """Test that invalid detection names raise ValueError."""
        detection = {
            "sample_token": "abc123",
            "translation": [1.0, 2.0, 3.0],
            "size": [1.5, 2.0, 1.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0],
            "detection_name": "invalid_class",  # Not in NUSCENES_DETECTION_CLASSES
            "detection_score": 0.9,
            "attribute_name": ""
        }
        submission = create_nuscenes_submission({"abc123": [detection]})

        with pytest.raises(ValueError, match="Invalid detection_name"):
            validate_submission_schema(submission)

    def test_wrong_translation_length_raises(self):
        """Test that wrong array lengths raise ValueError."""
        detection = {
            "sample_token": "abc123",
            "translation": [1.0, 2.0],  # Should be 3 elements
            "size": [1.5, 2.0, 1.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0],
            "detection_name": "car",
            "detection_score": 0.9,
            "attribute_name": ""
        }
        submission = create_nuscenes_submission({"abc123": [detection]})

        with pytest.raises(ValueError, match="translation must have 3"):
            validate_submission_schema(submission)


class TestClassNameNormalization:
    """Tests for class name normalization."""

    @pytest.mark.parametrize("label,expected", [
        ("car", "car"),
        ("Car", "car"),
        ("CAR", "car"),
        ("vehicle.car", "car"),
        ("human.pedestrian.adult", "pedestrian"),
        ("movable_object.barrier", "barrier"),
        ("truck", "truck"),
        ("bicycle", "bicycle"),
    ])
    def test_normalize_valid_labels(self, label, expected):
        """Test normalization of valid labels."""
        result = normalize_class_name(label)
        assert result == expected

    def test_unknown_label_returns_none(self):
        """Test that unknown labels return None."""
        result = normalize_class_name("unknown_class_xyz")
        assert result is None


class TestQuaternionConversion:
    """Tests for yaw to quaternion conversion."""

    def test_zero_yaw(self):
        """Test quaternion for zero yaw (identity rotation)."""
        quat = yaw_to_quaternion(0.0)
        assert len(quat) == 4
        assert np.isclose(quat[0], 1.0)  # w = cos(0) = 1
        assert np.isclose(quat[3], 0.0)  # z = sin(0) = 0

    def test_90_degree_yaw(self):
        """Test quaternion for 90 degree yaw."""
        quat = yaw_to_quaternion(np.pi / 2)
        assert len(quat) == 4
        assert np.isclose(quat[0], np.cos(np.pi / 4))  # w = cos(45)
        assert np.isclose(quat[3], np.sin(np.pi / 4))  # z = sin(45)

    def test_quaternion_is_unit(self):
        """Test that quaternion is unit length."""
        quat = yaw_to_quaternion(1.234)
        norm = np.sqrt(sum(q**2 for q in quat))
        assert np.isclose(norm, 1.0)


class TestBox3DConversion:
    """Tests for Box3D to nuScenes detection conversion."""

    def test_conversion_all_fields_present(self):
        """Test that conversion includes all required fields."""
        box = Box3D(
            center=np.array([1.0, 2.0, 3.0]),
            size=np.array([1.5, 2.0, 1.0]),
            yaw=0.5,
            label="car",
            score=0.9
        )
        ego_pose = np.eye(4)

        result = box3d_to_nuscenes_detection(box, "token123", ego_pose)

        required_keys = [
            "sample_token", "translation", "size", "rotation",
            "velocity", "detection_name", "detection_score", "attribute_name"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_invalid_class_returns_none(self):
        """Test that invalid class returns None."""
        box = Box3D(
            center=np.array([1.0, 2.0, 3.0]),
            size=np.array([1.5, 2.0, 1.0]),
            yaw=0.5,
            label="unknown_class",
            score=0.9
        )
        ego_pose = np.eye(4)

        result = box3d_to_nuscenes_detection(box, "token123", ego_pose)
        assert result is None

    def test_ego_pose_transforms_center(self):
        """Test that ego pose correctly transforms box center."""
        box = Box3D(
            center=np.array([1.0, 0.0, 0.0]),
            size=np.array([1.5, 2.0, 1.0]),
            yaw=0.0,
            label="car",
            score=0.9
        )
        # Ego pose with translation
        ego_pose = np.eye(4)
        ego_pose[0, 3] = 10.0  # Translate 10m in x
        ego_pose[1, 3] = 5.0   # Translate 5m in y

        result = box3d_to_nuscenes_detection(box, "token123", ego_pose)

        # Center should be transformed
        assert np.isclose(result["translation"][0], 11.0)  # 1 + 10
        assert np.isclose(result["translation"][1], 5.0)   # 0 + 5


class TestJSONSerialization:
    """Tests for JSON serialization of submissions."""

    def test_submission_is_json_serializable(self):
        """Test that submission can be serialized to JSON."""
        detection = {
            "sample_token": "abc123",
            "translation": [1.0, 2.0, 3.0],
            "size": [1.5, 2.0, 1.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0],
            "detection_name": "car",
            "detection_score": 0.9,
            "attribute_name": ""
        }
        submission = create_nuscenes_submission({"abc123": [detection]})

        # Should not raise
        json_str = json.dumps(submission)
        assert isinstance(json_str, str)

        # Should round-trip
        reloaded = json.loads(json_str)
        assert reloaded == submission


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
