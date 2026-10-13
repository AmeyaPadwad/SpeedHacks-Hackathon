"""Pette's local cache and world model.

`store.Store` is an append-only SQLite log of small facts (event names, timestamps, which kind of SOMA
suggestion was shown and what you did with it). `world_model.WorldModel` derives what Pette uses from it.
Never stored: suggestion text, screenshots, camera frames or landmarks, file contents or paths.
"""
from .store import Store, default_path
from .world_model import WorldModel

__all__ = ["Store", "WorldModel", "default_path"]
