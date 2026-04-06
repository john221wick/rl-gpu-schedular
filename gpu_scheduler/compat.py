from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, ConfigDict  # type: ignore
except ImportError:
    ConfigDict = dict

    class BaseModel:
        def __init__(self, **data: Any) -> None:
            annotations = getattr(self.__class__, "__annotations__", {})
            for field_name in annotations:
                if field_name in data:
                    setattr(self, field_name, data[field_name])
                elif hasattr(self.__class__, field_name):
                    setattr(self, field_name, getattr(self.__class__, field_name))
                else:
                    setattr(self, field_name, None)

        @classmethod
        def model_validate(cls, obj: Any) -> "BaseModel":
            if isinstance(obj, cls):
                return obj
            if isinstance(obj, dict):
                return cls(**obj)
            raise TypeError(f"Cannot validate object for {cls.__name__}: {type(obj)!r}")

        def model_dump(self) -> dict[str, Any]:
            annotations = getattr(self.__class__, "__annotations__", {})
            return {field_name: getattr(self, field_name) for field_name in annotations}
