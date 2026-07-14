"""Stage 4 on-policy rollout, PPO update, and checkpoint public interfaces."""

from .checkpoint import CheckpointManager
from .resume import (
    TrainingResumeReceipt,
    optimizer_state_sha256,
    resume_training_from_last_checkpoint,
)
from .rollout import CandidateSnapshot, RolloutBatch, RolloutBuffer, RolloutTransition
from .trainer import PPOTrainer, PPOUpdateMetrics

__all__ = [
    "CandidateSnapshot",
    "CheckpointManager",
    "PPOTrainer",
    "PPOUpdateMetrics",
    "RolloutBatch",
    "RolloutBuffer",
    "RolloutTransition",
    "TrainingResumeReceipt",
    "optimizer_state_sha256",
    "resume_training_from_last_checkpoint",
]
