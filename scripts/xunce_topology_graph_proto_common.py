from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TopologyGraphProtoOutput:
    logits: torch.Tensor
    masked_logits: torch.Tensor
    action_probs: torch.Tensor
    value: torch.Tensor


class TopologyAwareCoverageGraphPrototype(nn.Module):
    architecture_name = "topology_aware_coverage_graph_proto_v1"

    def __init__(
        self,
        *,
        candidate_feature_count: int,
        edge_feature_count: int,
        memory_feature_count: int,
        hidden_dim: int = 32,
        message_passing_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if candidate_feature_count <= 0:
            raise ValueError("candidate_feature_count must be positive")
        if edge_feature_count <= 0:
            raise ValueError("edge_feature_count must be positive")
        if memory_feature_count <= 0:
            raise ValueError("memory_feature_count must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if message_passing_layers <= 0:
            raise ValueError("message_passing_layers must be positive")
        self.candidate_feature_count = int(candidate_feature_count)
        self.edge_feature_count = int(edge_feature_count)
        self.memory_feature_count = int(memory_feature_count)
        self.hidden_dim = int(hidden_dim)
        self.message_passing_layers = int(message_passing_layers)
        self.dropout = float(dropout)

        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_feature_count, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_feature_count, self.hidden_dim),
            nn.GELU(),
        )
        self.memory_encoder = nn.Sequential(
            nn.Linear(self.memory_feature_count, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.message_layers = nn.ModuleList(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim) for _ in range(self.message_passing_layers)
        )
        self.message_norms = nn.ModuleList(nn.LayerNorm(self.hidden_dim) for _ in range(self.message_passing_layers))
        self.dropout_layer = nn.Dropout(self.dropout)
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        *,
        candidate_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        memory_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> TopologyGraphProtoOutput:
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [batch, candidates, features]")
        if edge_features.ndim != 2:
            raise ValueError("edge_features must have shape [edges, features]")
        if edge_index.ndim != 2 or edge_index.shape[1] != 2:
            raise ValueError("edge_index must have shape [edges, 2]")
        if memory_features.ndim != 2:
            raise ValueError("memory_features must have shape [batch, features]")
        if action_mask.ndim != 2 or action_mask.shape != candidate_features.shape[:2]:
            raise ValueError("action_mask must have shape [batch, candidates]")
        mask = action_mask.bool()
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each batch item must contain at least one valid action")
        if candidate_features.shape[2] != self.candidate_feature_count:
            raise ValueError("candidate feature count does not match prototype metadata")
        if edge_features.shape[1] != self.edge_feature_count:
            raise ValueError("edge feature count does not match prototype metadata")
        if memory_features.shape[1] != self.memory_feature_count:
            raise ValueError("memory feature count does not match prototype metadata")

        candidate_embedding = self.candidate_encoder(candidate_features)
        edge_embedding = self.edge_encoder(edge_features)
        candidate_embedding = self._message_pass(candidate_embedding, edge_embedding=edge_embedding, edge_index=edge_index)
        memory_embedding = self.memory_encoder(memory_features)
        expanded_memory = memory_embedding.unsqueeze(1).expand(-1, candidate_embedding.shape[1], -1)

        logits = self.policy_head(torch.cat((candidate_embedding, expanded_memory), dim=-1)).squeeze(-1)
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        action_probs = torch.softmax(masked_logits, dim=-1)

        masked_embedding = candidate_embedding * mask.unsqueeze(-1)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(candidate_embedding.dtype)
        pooled_candidates = masked_embedding.sum(dim=1) / valid_counts
        value = self.value_head(torch.cat((pooled_candidates, memory_embedding), dim=-1)).squeeze(-1)
        return TopologyGraphProtoOutput(
            logits=logits,
            masked_logits=masked_logits,
            action_probs=action_probs,
            value=value,
        )

    def _message_pass(
        self,
        candidate_embedding: torch.Tensor,
        *,
        edge_embedding: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return candidate_embedding
        left = edge_index[:, 0].long()
        right = edge_index[:, 1].long()
        candidate_count = candidate_embedding.shape[1]
        if int(left.min()) < 0 or int(right.min()) < 0 or int(left.max()) >= candidate_count or int(right.max()) >= candidate_count:
            raise ValueError("edge_index contains candidate index outside the candidate range")
        current = candidate_embedding
        for layer, norm in zip(self.message_layers, self.message_norms):
            messages = torch.zeros_like(current)
            degree = torch.zeros(
                (*current.shape[:2], 1),
                dtype=current.dtype,
                device=current.device,
            )
            for batch_index in range(current.shape[0]):
                left_embeddings = current[batch_index, left]
                right_embeddings = current[batch_index, right]
                left_messages = layer(torch.cat((right_embeddings, edge_embedding), dim=-1))
                right_messages = layer(torch.cat((left_embeddings, edge_embedding), dim=-1))
                messages[batch_index].index_add_(0, left, left_messages)
                messages[batch_index].index_add_(0, right, right_messages)
                ones = torch.ones((edge_index.shape[0], 1), dtype=current.dtype, device=current.device)
                degree[batch_index].index_add_(0, left, ones)
                degree[batch_index].index_add_(0, right, ones)
            aggregated = messages / degree.clamp_min(1.0)
            current = norm(current + self.dropout_layer(torch.nn.functional.gelu(aggregated)))
        return current


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
