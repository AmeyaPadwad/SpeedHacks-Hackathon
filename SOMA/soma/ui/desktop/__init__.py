"""SOMA consent-first desktop UI (PySide6).

Modules:

* ``app``: orchestration (consent dialog -> runtime -> bridge -> window/tray) and ``main``.
* ``bridge``: ``RuntimeBridge`` re-emits scheduler results on the Qt main thread and
  drops results that arrive after a pause; ``RuntimeLike`` is the runtime protocol.
* ``consent_dialog``, ``main_window``, ``suggestion_card``, ``score_chart``,
  ``memory_viewer``, ``debug_overlay`` (+ ``FrameTap``), ``tray``, ``theme``.

Launch with ``python -m scripts.run_desktop`` from ``SOMA/``.
"""
