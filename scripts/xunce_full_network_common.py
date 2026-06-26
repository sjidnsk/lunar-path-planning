from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

try:  # pragma: no cover
    from xunce_continuous_theta_action import theta_distribution_parameters
except ModuleNotFoundError:  # pragma: no cover
    from scripts.xunce_continuous_theta_action import theta_distribution_parameters


COMPATIBLE_MISSING_THETA_HEAD_PREFIXES = ("theta_mu_head.", "theta_kappa_head.")


@dataclass(frozen=True)
class XunceFullNetworkOutput:
    logits: torch.Tensor
    masked_logits: torch.Tensor
    action_probs: torch.Tensor
    value: torch.Tensor
    theta_mu_sin: torch.Tensor
    theta_mu_cos: torch.Tensor
    theta_kappa_raw: torch.Tensor
    theta_mu_rad: torch.Tensor
    theta_kappa: torch.Tensor


class XunceFullNetworkV1(nn.Module):
    architecture_name = "xunce_full_network_v1"

    def __init__(
        self,
        *,
        candidate_feature_count: int,
        edge_feature_count: int,
        memory_feature_count: int,
        context_feature_count: int,
        missing_indicator_count: int = 0,
        hidden_dim: int = 64,
        message_passing_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("candidate_feature_count", candidate_feature_count),
            ("edge_feature_count", edge_feature_count),
            ("memory_feature_count", memory_feature_count),
            ("context_feature_count", context_feature_count),
            ("hidden_dim", hidden_dim),
            ("message_passing_layers", message_passing_layers),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(missing_indicator_count) < 0:
            raise ValueError("missing_indicator_count must be non-negative")
        self.candidate_feature_count = int(candidate_feature_count)
        self.edge_feature_count = int(edge_feature_count)
        self.memory_feature_count = int(memory_feature_count)
        self.context_feature_count = int(context_feature_count)
        self.missing_indicator_count = int(missing_indicator_count)
        self.hidden_dim = int(hidden_dim)
        self.message_passing_layers = int(message_passing_layers)
        self.dropout = float(dropout)

        candidate_input_count = self.candidate_feature_count + self.missing_indicator_count
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_input_count, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_feature_count, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_feature_count, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.memory_encoder = nn.Sequential(
            nn.Linear(self.memory_feature_count, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.message_layers = nn.ModuleList(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim) for _ in range(self.message_passing_layers)
        )
        self.message_gates = nn.ModuleList(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim) for _ in range(self.message_passing_layers)
        )
        self.message_norms = nn.ModuleList(nn.LayerNorm(self.hidden_dim) for _ in range(self.message_passing_layers))
        self.topology_bias = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.theta_mu_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 2),
        )
        self.theta_kappa_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.dropout_layer = nn.Dropout(self.dropout)

    def metadata(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture_name,
            "candidate_feature_count": self.candidate_feature_count,
            "edge_feature_count": self.edge_feature_count,
            "memory_feature_count": self.memory_feature_count,
            "context_feature_count": self.context_feature_count,
            "missing_indicator_count": self.missing_indicator_count,
            "hidden_dim": self.hidden_dim,
            "message_passing_layers": self.message_passing_layers,
            "dropout": self.dropout,
            "candidate_graph_encoder_used": True,
            "topology_bias_used": True,
            "coverage_memory_token_used": True,
            "roi_budget_fusion_used": True,
            "masked_logits_head_used": True,
            "continuous_theta_head_available": True,
            "continuous_theta_distribution": "von_mises",
            "value_head_used": True,
            "production_registered": False,
        }

    def forward(
        self,
        *,
        candidate_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        memory_features: torch.Tensor,
        context_features: torch.Tensor,
        action_mask: torch.Tensor,
        candidate_missing_indicators: torch.Tensor | None = None,
    ) -> XunceFullNetworkOutput:
        self._validate_inputs(
            candidate_features=candidate_features,
            edge_features=edge_features,
            edge_index=edge_index,
            memory_features=memory_features,
            context_features=context_features,
            action_mask=action_mask,
            candidate_missing_indicators=candidate_missing_indicators,
        )
        if candidate_missing_indicators is None:
            candidate_missing_indicators = torch.zeros(
                (*candidate_features.shape[:2], self.missing_indicator_count),
                dtype=candidate_features.dtype,
                device=candidate_features.device,
            )
        candidate_inputs = torch.cat((candidate_features, candidate_missing_indicators.to(candidate_features.dtype)), dim=-1)
        candidate_embedding = self.candidate_encoder(candidate_inputs)
        context_embedding = self.context_encoder(context_features)
        memory_embedding = self.memory_encoder(memory_features)
        edge_embedding = self.edge_encoder(edge_features)

        graph_embedding, topology_node_bias = self._message_pass(
            candidate_embedding + context_embedding,
            edge_embedding=edge_embedding,
            edge_index=edge_index,
            memory_embedding=memory_embedding,
        )
        expanded_memory = memory_embedding.unsqueeze(1).expand(-1, graph_embedding.shape[1], -1)
        fused = torch.cat((graph_embedding, context_embedding, expanded_memory), dim=-1)
        logits = self.policy_head(fused).squeeze(-1) + topology_node_bias.squeeze(-1)
        theta_mu = self.theta_mu_head(fused)
        theta_mu_sin = theta_mu[..., 0]
        theta_mu_cos = theta_mu[..., 1]
        theta_kappa_raw = self.theta_kappa_head(fused).squeeze(-1)
        theta_mu_rad, theta_kappa = theta_distribution_parameters(theta_mu_sin, theta_mu_cos, theta_kappa_raw)
        mask = action_mask.bool()
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        action_probs = torch.softmax(masked_logits, dim=-1)

        pooled_graph = (graph_embedding * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(graph_embedding.dtype)
        pooled_context = (context_embedding * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(context_embedding.dtype)
        value = self.value_head(torch.cat((pooled_graph, pooled_context, memory_embedding), dim=-1)).squeeze(-1)
        return XunceFullNetworkOutput(
            logits=logits,
            masked_logits=masked_logits,
            action_probs=action_probs,
            value=value,
            theta_mu_sin=theta_mu_sin,
            theta_mu_cos=theta_mu_cos,
            theta_kappa_raw=theta_kappa_raw,
            theta_mu_rad=theta_mu_rad,
            theta_kappa=theta_kappa,
        )

    def _message_pass(
        self,
        candidate_embedding: torch.Tensor,
        *,
        edge_embedding: torch.Tensor,
        edge_index: torch.Tensor,
        memory_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = candidate_embedding
        topology_node_bias = torch.zeros((*current.shape[:2], 1), dtype=current.dtype, device=current.device)
        topology_degree = torch.zeros((*current.shape[:2], 1), dtype=current.dtype, device=current.device)
        if edge_index.numel() == 0:
            return current, topology_node_bias
        left = edge_index[:, 0].long()
        right = edge_index[:, 1].long()
        candidate_count = current.shape[1]
        if int(left.min()) < 0 or int(right.min()) < 0 or int(left.max()) >= candidate_count or int(right.max()) >= candidate_count:
            raise ValueError("edge_index contains candidate index outside the candidate range")
        edge_bias = self.topology_bias(edge_embedding)
        edge_bias_ones = torch.ones((edge_index.shape[0], 1), dtype=current.dtype, device=current.device)
        for batch_index in range(current.shape[0]):
            topology_node_bias[batch_index].index_add_(0, left, edge_bias)
            topology_node_bias[batch_index].index_add_(0, right, edge_bias)
            topology_degree[batch_index].index_add_(0, left, edge_bias_ones)
            topology_degree[batch_index].index_add_(0, right, edge_bias_ones)
        for layer, gate_layer, norm in zip(self.message_layers, self.message_gates, self.message_norms):
            messages = torch.zeros_like(current)
            degree = torch.zeros((*current.shape[:2], 1), dtype=current.dtype, device=current.device)
            for batch_index in range(current.shape[0]):
                memory_for_edges = memory_embedding[batch_index].unsqueeze(0).expand(edge_embedding.shape[0], -1)
                left_embeddings = current[batch_index, left]
                right_embeddings = current[batch_index, right]
                gate = torch.sigmoid(gate_layer(torch.cat((edge_embedding, memory_for_edges), dim=-1)))
                left_messages = layer(torch.cat((right_embeddings, edge_embedding, memory_for_edges), dim=-1)) * gate
                right_messages = layer(torch.cat((left_embeddings, edge_embedding, memory_for_edges), dim=-1)) * gate
                messages[batch_index].index_add_(0, left, left_messages)
                messages[batch_index].index_add_(0, right, right_messages)
                ones = torch.ones((edge_index.shape[0], 1), dtype=current.dtype, device=current.device)
                degree[batch_index].index_add_(0, left, ones)
                degree[batch_index].index_add_(0, right, ones)
            aggregated = messages / degree.clamp_min(1.0)
            current = norm(current + self.dropout_layer(torch.nn.functional.gelu(aggregated)))
        topology_node_bias = topology_node_bias / topology_degree.clamp_min(1.0)
        return current, topology_node_bias

    def _validate_inputs(
        self,
        *,
        candidate_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        memory_features: torch.Tensor,
        context_features: torch.Tensor,
        action_mask: torch.Tensor,
        candidate_missing_indicators: torch.Tensor | None,
    ) -> None:
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [batch, candidates, features]")
        if candidate_features.shape[2] != self.candidate_feature_count:
            raise ValueError("candidate feature count does not match metadata")
        if edge_features.ndim != 2 or edge_features.shape[1] != self.edge_feature_count:
            raise ValueError("edge_features must have shape [edges, edge_feature_count]")
        if edge_index.ndim != 2 or edge_index.shape[1] != 2:
            raise ValueError("edge_index must have shape [edges, 2]")
        if memory_features.ndim != 2 or memory_features.shape[1] != self.memory_feature_count:
            raise ValueError("memory_features must have shape [batch, memory_feature_count]")
        if context_features.ndim != 3 or context_features.shape[:2] != candidate_features.shape[:2] or context_features.shape[2] != self.context_feature_count:
            raise ValueError("context_features must have shape [batch, candidates, context_feature_count]")
        if action_mask.ndim != 2 or action_mask.shape != candidate_features.shape[:2]:
            raise ValueError("action_mask must have shape [batch, candidates]")
        if not torch.all(action_mask.bool().any(dim=1)):
            raise ValueError("each batch item must contain at least one valid action")
        if candidate_missing_indicators is not None:
            if candidate_missing_indicators.ndim != 3:
                raise ValueError("candidate_missing_indicators must have shape [batch, candidates, features]")
            if candidate_missing_indicators.shape[:2] != candidate_features.shape[:2]:
                raise ValueError("candidate_missing_indicators must match candidate batch and count")
            if candidate_missing_indicators.shape[2] != self.missing_indicator_count:
                raise ValueError("candidate_missing_indicators feature count does not match metadata")


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def load_xunce_full_network_state_dict_compatible(
    model: XunceFullNetworkV1,
    state_dict: dict[str, Any],
) -> tuple[bool, list[str], list[str], list[str]]:
    """Allow only newly-added continuous-theta heads to be missing from legacy checkpoints."""
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    disallowed_missing = [
        key for key in missing if not key.startswith(COMPATIBLE_MISSING_THETA_HEAD_PREFIXES)
    ]
    return not disallowed_missing and not unexpected, missing, disallowed_missing, unexpected
