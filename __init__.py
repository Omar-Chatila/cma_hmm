"""Standalone HMM state annotation helpers for movement trajectories."""

from importlib import import_module

_EXPORTS = {
    "ColumnConfig": "preprocessing",
    "HMMStateAnnotator": "state_annotation",
    "apply_moveapps_id_dtype_patch": "move_apps_patch",
    "annotate_states" : "state_annotation",
    "angle_diff": "utils",
    "annotate_states_gdf": "state_annotation",
    "apply_hmm": "models",
    "detect_typical_interval": "utils",
    "merge_states_to_gdf": "utils",
    "preprocess_hmm": "preprocessing",
    "process_trajectories": "preprocessing",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value
