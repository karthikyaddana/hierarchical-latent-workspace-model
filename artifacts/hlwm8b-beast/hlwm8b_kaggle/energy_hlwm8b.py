from __future__ import annotations

import subprocess
import threading
import time
from typing import Dict, List


class EnergyMeter:
    """Integrate NVIDIA power samples. This is a run-level estimate, not lab instrumentation."""

    def __init__(self, interval_seconds: float = 5.0) -> None:
        self.interval_seconds = float(interval_seconds)
        self.samples: List[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _power_watts() -> float:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return sum(float(line.strip()) for line in result.stdout.splitlines() if line.strip())

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.monotonic(), self._power_watts()))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1)
        watt_seconds = 0.0
        for (first_time, first_power), (second_time, second_power) in zip(
            self.samples, self.samples[1:]
        ):
            watt_seconds += (second_time - first_time) * (first_power + second_power) / 2.0
        elapsed = self.samples[-1][0] - self.samples[0][0] if len(self.samples) >= 2 else 0.0
        return {
            "samples": float(len(self.samples)),
            "elapsed_seconds": elapsed,
            "energy_wh": watt_seconds / 3600.0,
            "mean_power_w": watt_seconds / elapsed if elapsed > 0 else 0.0,
        }
