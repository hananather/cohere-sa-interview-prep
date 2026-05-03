from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter("defence_agent_requests_total", "Total API requests", ["route", "status"])
REQUEST_LATENCY = Histogram("defence_agent_request_latency_seconds", "Request latency", ["route"])
ROUTE_COUNT = Counter("defence_agent_route_total", "Selected agent routes", ["route"])
TOOL_COUNT = Counter("defence_agent_tool_calls_total", "Tool calls", ["tool", "status"])
ERROR_COUNT = Counter("defence_agent_errors_total", "Errors", ["component"])
SAFETY_BLOCKS = Counter("defence_agent_safety_blocks_total", "Safety blocks", ["stage"])
UNAUTHORIZED_ATTEMPTS = Counter("defence_agent_unauthorized_attempts_total", "Unauthorized attempts", ["resource"])
