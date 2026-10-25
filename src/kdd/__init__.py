"""Paquete de soporte KDD: registro de fases y traducción a conocimiento."""

from .phase_logger import KDDLogger, KDD_PHASES, Phase
from .knowledge import (
    SEGMENT_PLAYBOOK,
    build_knowledge_payload,
    build_playbook_markdown,
    playbook_for,
)

__all__ = [
    "KDDLogger",
    "KDD_PHASES",
    "Phase",
    "SEGMENT_PLAYBOOK",
    "build_knowledge_payload",
    "build_playbook_markdown",
    "playbook_for",
]
