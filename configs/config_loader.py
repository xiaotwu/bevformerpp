"""Configuration loader for BEV Fusion System."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """Configuration class that allows dot notation access to nested dictionaries."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self._config = config_dict
        
        # Convert nested dictionaries to Config objects
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        self._config[key] = value
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration back to dictionary."""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._config})"


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged


def load_config(config_path: Union[str, Path], base_config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from YAML file with optional base configuration.
    
    Args:
        config_path: Path to main configuration file
        base_config_path: Optional path to base configuration file
        
    Returns:
        Config object with loaded configuration
        
    Example:
        >>> config = load_config("configs/train_config.yaml")
        >>> print(config.training.batch_size)
        4
        >>> print(config.model.lidar.num_features)
        64
    """
    config_path = Path(config_path)
    
    # Load main configuration
    config_dict = load_yaml(config_path)
    
    # Check if configuration references a base configuration
    if "base" in config_dict:
        base_ref = config_dict.pop("base")
        
        # Resolve base configuration path relative to current config
        if base_config_path is None:
            base_config_path = config_path.parent / base_ref
        
        # Load base configuration
        base_dict = load_yaml(base_config_path)
        
        # Merge configurations (config_dict overrides base_dict)
        config_dict = merge_configs(base_dict, config_dict)
    elif base_config_path is not None:
        # Explicit base configuration provided
        base_dict = load_yaml(base_config_path)
        config_dict = merge_configs(base_dict, config_dict)
    
    return Config(config_dict)


def save_config(config: Union[Config, Dict[str, Any]], save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object or dictionary to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Config to dictionary if needed
    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def parse_override(override_str: str) -> tuple:
    """Parse a single CLI override string.

    Args:
        override_str: String in format "key.subkey=value" or "key.subkey.subsubkey=value"

    Returns:
        Tuple of (key_parts, value) where key_parts is a list of nested keys

    Examples:
        >>> parse_override("training.batch_size=4")
        (['training', 'batch_size'], 4)
        >>> parse_override("model.fusion.type=local_attention")
        (['model', 'fusion', 'type'], 'local_attention')
    """
    if '=' not in override_str:
        raise ValueError(f"Override must be in format 'key=value', got: {override_str}")

    key_path, value_str = override_str.split('=', 1)
    key_parts = key_path.split('.')

    # Try to parse value as appropriate type
    value = _parse_value(value_str)

    return key_parts, value


def _parse_value(value_str: str) -> Any:
    """Parse a value string to the appropriate Python type.

    Args:
        value_str: String representation of the value

    Returns:
        Parsed value (int, float, bool, None, list, or str)
    """
    value_str = value_str.strip()

    # Handle None
    if value_str.lower() in ('none', 'null'):
        return None

    # Handle booleans
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False

    # Handle lists (simple comma-separated)
    if value_str.startswith('[') and value_str.endswith(']'):
        inner = value_str[1:-1].strip()
        if not inner:
            return []
        items = [_parse_value(item.strip()) for item in inner.split(',')]
        return items

    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string (strip quotes if present)
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]

    return value_str


def apply_overrides(config_dict: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    """Apply CLI overrides to configuration dictionary.

    Args:
        config_dict: Configuration dictionary to modify
        overrides: List of override strings in format "key.subkey=value"

    Returns:
        Modified configuration dictionary

    Example:
        >>> config = {'training': {'batch_size': 2, 'epochs': 10}}
        >>> apply_overrides(config, ['training.batch_size=4', 'training.epochs=20'])
        {'training': {'batch_size': 4, 'epochs': 20}}
    """
    result = config_dict.copy()

    for override_str in overrides:
        key_parts, value = parse_override(override_str)

        # Navigate to the nested location
        current = result
        for i, key in enumerate(key_parts[:-1]):
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # If we hit a non-dict before the last key, make it a dict
                current[key] = {}
            current = current[key]

        # Set the final value
        final_key = key_parts[-1]
        current[final_key] = value

    return result


def load_config_with_overrides(
    config_path: Union[str, Path],
    overrides: Optional[list] = None,
    base_config_path: Optional[Union[str, Path]] = None
) -> Config:
    """Load configuration with CLI override support.

    Args:
        config_path: Path to main configuration file
        overrides: List of override strings in format "key.subkey=value"
        base_config_path: Optional path to base configuration file

    Returns:
        Config object with loaded and overridden configuration

    Example:
        >>> config = load_config_with_overrides(
        ...     "configs/train_config.yaml",
        ...     overrides=["training.batch_size=4", "training.epochs=50"]
        ... )
    """
    config_path = Path(config_path)

    # Load main configuration
    config_dict = load_yaml(config_path)

    # Handle base config
    if "base" in config_dict:
        base_ref = config_dict.pop("base")
        if base_config_path is None:
            base_config_path = config_path.parent / base_ref
        base_dict = load_yaml(base_config_path)
        config_dict = merge_configs(base_dict, config_dict)
    elif base_config_path is not None:
        base_dict = load_yaml(base_config_path)
        config_dict = merge_configs(base_dict, config_dict)

    # Apply CLI overrides
    if overrides:
        config_dict = apply_overrides(config_dict, overrides)

    return Config(config_dict)
