from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    name: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def gemini_declarations(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def run(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(name=name, ok=False, error="Unknown tool")
        payload = arguments or {}
        if not isinstance(payload, dict):
            return ToolResult(name=name, ok=False, error="Tool arguments must be an object")
        try:
            return tool.execute(**payload)
        except TypeError:
            return ToolResult(name=name, ok=False, error="Invalid tool arguments")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(name=name, ok=False, error=str(exc))
