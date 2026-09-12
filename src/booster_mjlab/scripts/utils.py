"""Utilities shared by script entry points."""

import pickle
from typing import Any


def remove_unpicklable_objects(obj: Any, path: str = "root") -> Any:
    """Recursively remove unpicklable objects from a nested structure."""
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            cleaned_value = remove_unpicklable_objects(value, f"{path}.{key}")
            try:
                # Recurse first so only truly unpicklable leaves are removed.
                pickle.dumps(cleaned_value)
                cleaned[key] = cleaned_value
            except (TypeError, AttributeError, pickle.PicklingError) as e:
                print(
                    f"[WARNING] Removing unpicklable object at {path}.{key}: "
                    f"{type(cleaned_value).__name__} - {e}"
                )
        return cleaned

    if isinstance(obj, (list, tuple)):
        cleaned = []
        for i, item in enumerate(obj):
            cleaned_item = remove_unpicklable_objects(item, f"{path}[{i}]")
            try:
                # Recurse first so only truly unpicklable leaves are removed.
                pickle.dumps(cleaned_item)
                cleaned.append(cleaned_item)
            except (TypeError, AttributeError, pickle.PicklingError) as e:
                print(
                    f"[WARNING] Removing unpicklable object at {path}[{i}]: "
                    f"{type(cleaned_item).__name__} - {e}"
                )
        return type(obj)(cleaned) if isinstance(obj, tuple) else cleaned

    return obj
