"""
KDD phase logger.

Records the input, output, metrics, decisions and warnings of every phase of
the Knowledge Discovery in Databases process. The result is exported as a
machine-readable JSON file and as a Markdown report that can be attached to the
academic document.

Usage
-----
>>> logger = KDDLogger()
>>> with logger.phase(3, "Preprocesamiento") as phase:
...     phase.metric("filas_entrada", len(df))
...     phase.decision("Se eliminan duplicados exactos")
>>> logger.export()
"""

from __future__ import annotations

import json
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.settings import KDD_OUTPUT_DIR

# Canonical names of the seven KDD phases (Fayyad et al., 1996)
KDD_PHASES = {
    1: "Comprensión del dominio y definición del objetivo",
    2: "Selección de datos",
    3: "Preprocesamiento y limpieza",
    4: "Transformación y reducción",
    5: "Minería de datos",
    6: "Interpretación y evaluación",
    7: "Presentación y uso del conocimiento",
}


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars and containers into JSON-safe objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class Phase:
    """Mutable record for a single KDD phase."""

    def __init__(self, number: int, title: Optional[str] = None):
        self.number = number
        self.title = title or KDD_PHASES.get(number, f"Fase {number}")
        self.status = "ok"
        self.metrics: Dict[str, Any] = {}
        self.decisions: List[str] = []
        self.warnings: List[str] = []
        self.error: Optional[str] = None
        self.duration_s = 0.0

    def metric(self, key: str, value: Any) -> "Phase":
        self.metrics[key] = _jsonable(value)
        return self

    def metrics_update(self, values: Dict[str, Any]) -> "Phase":
        for key, value in values.items():
            self.metric(key, value)
        return self

    def decision(self, text: str) -> "Phase":
        self.decisions.append(text)
        return self

    def warn(self, text: str) -> "Phase":
        self.warnings.append(text)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.number,
            "title": self.title,
            "status": self.status,
            "duration_s": round(self.duration_s, 3),
            "metrics": self.metrics,
            "decisions": self.decisions,
            "warnings": self.warnings,
            "error": self.error,
        }


class KDDLogger:
    """Collects the evidence produced by each KDD phase."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or KDD_OUTPUT_DIR)
        self.records: List[Phase] = []
        self._current: Optional[Phase] = None
        self.started_at = datetime.now()

    # ------------------------------------------------------------------ phases
    @contextmanager
    def phase(self, number: int, title: Optional[str] = None):
        """Context manager that times a phase and captures failures."""
        phase = Phase(number, title)
        self._current = phase
        start = time.perf_counter()
        try:
            yield phase
        except Exception as exc:
            phase.status = "error"
            phase.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            phase.duration_s = time.perf_counter() - start
            self.records.append(phase)
            self._current = None

    # ---------------------------------------------------------------- shortcuts
    def metric(self, key: str, value: Any) -> None:
        if self._current is not None:
            self._current.metric(key, value)

    def decision(self, text: str) -> None:
        if self._current is not None:
            self._current.decision(text)

    def warn(self, text: str) -> None:
        if self._current is not None:
            self._current.warn(text)

    # ----------------------------------------------------------------- exports
    def to_dict(self) -> Dict[str, Any]:
        phases = sorted(self.records, key=lambda p: p.number)
        return {
            "generated_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "phases": [p.to_dict() for p in phases],
        }

    def export(self, json_path: Optional[str] = None,
               markdown_path: Optional[str] = None) -> Dict[str, str]:
        """Write the JSON and Markdown reports and return their paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = Path(json_path or self.output_dir / "kdd_metrics.json")
        markdown_path = Path(markdown_path or self.output_dir / "kdd_report.md")

        payload = self.to_dict()
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(self.to_markdown(payload), encoding="utf-8")

        return {"json": str(json_path), "markdown": str(markdown_path)}

    def to_markdown(self, payload: Optional[Dict[str, Any]] = None) -> str:
        payload = payload or self.to_dict()
        env = payload["environment"]

        lines = [
            "# Reporte de ejecución del proceso KDD",
            "",
            f"Generado: {payload['generated_at']}  ",
            f"Python: {env['python']}  ",
            f"Plataforma: {env['platform']}",
            "",
            "Este archivo es evidencia automática de la aplicación de las siete "
            "fases del proceso KDD. Cada fase registra sus métricas, las decisiones "
            "tomadas y las advertencias detectadas.",
            "",
            "## Resumen de fases",
            "",
            "| Fase | Nombre | Estado | Duración (s) | Métricas | Alertas |",
            "|---|---|---|---|---|---|",
        ]

        for phase in payload["phases"]:
            lines.append(
                "| {phase} | {title} | {status} | {duration_s} | {n_metrics} | {n_warn} |".format(
                    phase=phase["phase"],
                    title=phase["title"],
                    status=phase["status"],
                    duration_s=phase["duration_s"],
                    n_metrics=len(phase["metrics"]),
                    n_warn=len(phase["warnings"]),
                )
            )

        for phase in payload["phases"]:
            lines += ["", f"## Fase {phase['phase']}. {phase['title']}", ""]
            lines.append(f"**Estado:** `{phase['status']}` · "
                         f"**Duración:** {phase['duration_s']} s")
            if phase.get("error"):
                lines += ["", f"> **Error:** {phase['error']}"]

            if phase["metrics"]:
                lines += ["", "### Métricas", ""]
                for key, value in phase["metrics"].items():
                    if isinstance(value, (dict, list)):
                        lines += [
                            f"- **{key}**",
                            "",
                            "```json",
                            json.dumps(value, ensure_ascii=False, indent=2),
                            "```",
                        ]
                    else:
                        lines.append(f"- **{key}:** {value}")

            if phase["decisions"]:
                lines += ["", "### Decisiones", ""]
                lines += [f"- {text}" for text in phase["decisions"]]

            if phase["warnings"]:
                lines += ["", "### Advertencias", ""]
                lines += [f"- {text}" for text in phase["warnings"]]

        lines.append("")
        return "\n".join(lines)
