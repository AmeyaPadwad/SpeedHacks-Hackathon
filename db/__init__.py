"""Pette's local cache and world model.

`store.Store` is an append-only SQLite log of small facts (event names, timestamps, recipient/platform,
approve/cancel outcomes). `world_model.WorldModel` derives what Pette uses from it. Never stored: message
text, screenshots, camera frames or landmarks, audio, file contents, API keys.
"""
from .store import Store, default_path
from .world_model import WorldModel

__all__ = ["Store", "WorldModel", "default_path"]
