from typing import Any

from app.agent.tools_base import BaseTool, ToolRegistry, ToolResult
from app.config.settings import get_settings
from app.utils.time import now_tz


class GetCurrentTimeTool(BaseTool):
    name = "get_current_time"
    description = (
        "Get the current timezone-aware date and time. Always use this instead of guessing "
        "when a visitor asks what time or date it is."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name. Defaults to Asia/Kolkata.",
            }
        },
    }

    def execute(self, timezone: str | None = None, **_: Any) -> ToolResult:
        settings = get_settings()
        tz_name = (timezone or settings.default_timezone).strip()
        if len(tz_name) > 64 or any(char in tz_name for char in ";|&`$"):
            return ToolResult(name=self.name, ok=False, error="Invalid timezone")
        moment = now_tz(tz_name)
        return ToolResult(
            name=self.name,
            ok=True,
            data={
                "datetime": moment.isoformat(),
                "date": moment.date().isoformat(),
                "time": moment.strftime("%H:%M:%S"),
                "timezone": str(moment.tzinfo),
            },
        )


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GetCurrentTimeTool())
    return registry
