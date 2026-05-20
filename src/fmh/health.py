from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class HealthResult:
    ok: bool
    detail: str


class VLLMHealthChecker:
    def __init__(self, timeout_sec: int, interval_sec: int) -> None:
        self.timeout_sec = timeout_sec
        self.interval_sec = interval_sec

    def wait_until_ready(self, endpoint: str) -> HealthResult:
        deadline = time.monotonic() + self.timeout_sec
        url = endpoint.rstrip("/") + "/v1/models"
        last_error = ""

        while time.monotonic() <= deadline:
            try:
                response = httpx.get(url, timeout=5)
                if response.status_code == 200:
                    return HealthResult(ok=True, detail=f"healthy: {url}")
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(self.interval_sec)

        return HealthResult(ok=False, detail=last_error or "health check timed out")
