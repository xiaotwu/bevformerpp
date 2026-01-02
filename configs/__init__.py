"""Configuration management for BEV Fusion System."""

from .config_loader import load_config, merge_configs, Config

__all__ = ["load_config", "merge_configs", "Config"]
