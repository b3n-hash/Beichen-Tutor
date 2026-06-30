"""Runtime performance instrumentation for the tutoring pipeline.

Diagnostic infrastructure only — records the wall-clock duration of significant
runtime operations so latency can be measured rather than guessed. Nothing here
changes tutoring behaviour.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingEvent:
    name: str
    start: float
    end: float
    duration_ms: float
    metadata: dict = field(default_factory=dict)

    @property
    def is_llm(self) -> bool:
        return self.metadata.get("model") is not None

    @property
    def is_umbrella(self) -> bool:
        # Umbrella events wrap other measured events (e.g. "Session Creation Total"
        # contains the analysis/conclusion/decomposition LLM calls) and are excluded
        # from summary sums to avoid double counting.
        return bool(self.metadata.get("umbrella"))


class PerformanceLogger:
    """Collects TimingEvents. Use `with perf.measure(name): ...` for wall-clock
    stages, or `perf.record_llm(...)` for LLM calls with token metadata. When a
    log_fn is attached, every completed timing also emits a concise [PERF] line."""

    def __init__(self, log_fn=None):
        self.events: list[TimingEvent] = []
        self._open: dict[str, float] = {}
        self._log_fn = log_fn  # optional callable(tag, message)

    # --- timing API -----------------------------------------------------------
    def start(self, name: str) -> None:
        self._open[name] = time.perf_counter()

    def stop(self, name: str, metadata: dict | None = None):
        start = self._open.pop(name, None)
        if start is None:
            return None
        return self._add(name, start, time.perf_counter(), dict(metadata or {}))

    @contextmanager
    def measure(self, name: str, metadata: dict | None = None):
        start = time.perf_counter()
        try:
            yield
        finally:  # close correctly even on exceptions
            self._add(name, start, time.perf_counter(), dict(metadata or {}))

    def record_llm(self, name: str, latency_ms: float, model=None, prompt_tokens=None,
                   completion_tokens=None, total_tokens=None, prompt_chars=None,
                   completion_chars=None):
        end = time.perf_counter()
        start = end - (latency_ms / 1000.0)
        metadata = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_chars": prompt_chars,
            "completion_chars": completion_chars,
        }
        return self._add(name, start, end, metadata, duration_ms=latency_ms)

    # --- internals ------------------------------------------------------------
    def _add(self, name, start, end, metadata, duration_ms=None) -> TimingEvent:
        if duration_ms is None:
            duration_ms = (end - start) * 1000.0
        event = TimingEvent(name=name, start=start, end=end, duration_ms=duration_ms, metadata=metadata)
        self.events.append(event)
        self._emit(event)
        return event

    def _emit(self, event: TimingEvent) -> None:
        if self._log_fn is None:
            return
        if event.is_llm:
            m = event.metadata
            msg = (f"{event.name} model={m.get('model')} latency={event.duration_ms:.0f} ms "
                   f"tokens={m.get('total_tokens')} prompt_chars={m.get('prompt_chars')} "
                   f"completion_chars={m.get('completion_chars')}")
        else:
            msg = f"{event.name} {_fmt_ms(event.duration_ms)}"
        self._log_fn("PERF", msg)

    # --- reporting ------------------------------------------------------------
    def summary(self) -> dict:
        non_umbrella = [e for e in self.events if not e.is_umbrella]
        llm = [e for e in self.events if e.is_llm]
        stage_durs = [e.duration_ms for e in non_umbrella]
        llm_durs = [e.duration_ms for e in llm]

        def _sum_tok(field):
            return sum((e.metadata.get(field) or 0) for e in llm)

        fastest = min(non_umbrella, key=lambda e: e.duration_ms) if non_umbrella else None
        slowest = max(non_umbrella, key=lambda e: e.duration_ms) if non_umbrella else None
        return {
            "fastest_stage": (fastest.name, fastest.duration_ms) if fastest else None,
            "slowest_stage": (slowest.name, slowest.duration_ms) if slowest else None,
            "avg_stage_ms": (sum(stage_durs) / len(stage_durs)) if stage_durs else 0.0,
            "avg_llm_ms": (sum(llm_durs) / len(llm_durs)) if llm_durs else 0.0,
            "total_llm_ms": sum(llm_durs),
            "total_runtime_ms": sum(stage_durs),
            "total_prompt_tokens": _sum_tok("prompt_tokens"),
            "total_completion_tokens": _sum_tok("completion_tokens"),
            "total_tokens": _sum_tok("total_tokens"),
            "llm_count": len(llm),
        }

    def format_report(self, width: int = 32) -> str:
        if not self.events:
            return "_(no timing data)_"

        lines = []
        llm_counter = 0
        for e in self.events:
            name = e.name
            if name == "LLM Request":
                llm_counter += 1
                name = f"LLM Request #{llm_counter}"
            lines.append(_leader(name, _fmt_ms(e.duration_ms), width))

        s = self.summary()
        lines.append("")
        if s["fastest_stage"]:
            lines.append(_leader("Fastest stage", f"{s['fastest_stage'][0]} ({_fmt_ms(s['fastest_stage'][1])})", width))
        if s["slowest_stage"]:
            lines.append(_leader("Slowest stage", f"{s['slowest_stage'][0]} ({_fmt_ms(s['slowest_stage'][1])})", width))
        lines.append(_leader("Average stage duration", _fmt_ms(s["avg_stage_ms"]), width))
        lines.append(_leader("Average LLM latency", _fmt_ms(s["avg_llm_ms"]), width))
        lines.append(_leader("Total LLM time", _fmt_dur(s["total_llm_ms"]), width))
        lines.append(_leader("Total runtime", _fmt_dur(s["total_runtime_ms"]), width))
        lines.append(_leader("Total prompt tokens", str(s["total_prompt_tokens"]), width))
        lines.append(_leader("Total completion tokens", str(s["total_completion_tokens"]), width))
        lines.append(_leader("Total tokens", str(s["total_tokens"]), width))
        return "\n".join(lines)


def _fmt_ms(ms: float) -> str:
    if ms < 10:
        return f"{ms:.1f} ms"
    return f"{ms:.0f} ms"


def _fmt_dur(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    return _fmt_ms(ms)


def _leader(label: str, value: str, width: int = 32) -> str:
    dots = "." * max(3, width - len(label))
    return f"{label} {dots} {value}"
