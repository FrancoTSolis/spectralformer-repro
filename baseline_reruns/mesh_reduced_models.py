"""Paper-faithful mesh-reduced graph and temporal model components.

The graph reducer/up-sampler follows Han et al. (ICLR 2022), equations
(2)--(5), with three non-residual GraphNet blocks and four latent values per
pivotal location.  The position-based variant follows Sun et al. (NeurIPS
2023), equations (5)--(9), including residual message passing, layer
normalization, and inverse-square k-nearest-neighbor interpolation.

PhysicsNeMo commit 77b3c68001159b948a16804fa76eb127735fb6d1 provides the
authoritative public PbGMR/GMUS spatial implementation.  The conditional flow
mirrors the authors' release at commit
383882bb56b33c48127a1874384e9897b5cde6e4 (alternating masks, conditional
scale/shift networks, bounded log scales, and invertible batch normalization),
configured with the paper's two coupling layers.  It is implemented locally
to remove the release's hard-coded devices and unrelated GluonTS dependency.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class MLP(nn.Module):
    """ReLU MLP with an explicit number of hidden layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int = 2,
        *,
        activation: type[nn.Module] = nn.ReLU,
        zero_last: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current_dim, hidden_dim), activation()))
            current_dim = hidden_dim
        final = nn.Linear(current_dim, output_dim)
        if zero_last:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GraphNetBlock(nn.Module):
    """One GraphNet processor block from the published update equations."""

    def __init__(self, width: int, *, residual_layernorm: bool) -> None:
        super().__init__()
        self.edge_model = MLP(3 * width, width, width, hidden_layers=2)
        self.node_model = MLP(2 * width, width, width, hidden_layers=2)
        self.residual_layernorm = residual_layernorm
        if residual_layernorm:
            self.edge_norm = nn.LayerNorm(width)
            self.node_norm = nn.LayerNorm(width)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sender, receiver = edge_index
        edge_update = self.edge_model(
            torch.cat((edges, nodes[:, sender], nodes[:, receiver]), dim=-1)
        )

        # Equations (3) in Han and (7) in Sun aggregate e^(l-1), not the
        # just-computed e^l.  Keeping that detail also distinguishes this
        # implementation from a MeshGraphNet processor.
        aggregate = torch.zeros(
            nodes.shape[0],
            nodes.shape[1],
            edges.shape[-1],
            dtype=edges.dtype,
            device=edges.device,
        )
        aggregate.index_add_(1, receiver, edges)
        node_update = self.node_model(torch.cat((nodes, aggregate), dim=-1))

        if self.residual_layernorm:
            edges = edges + self.edge_norm(edge_update)
            nodes = nodes + self.node_norm(node_update)
        else:
            edges = edge_update
            nodes = node_update
        return nodes, edges


class GraphMeshEncoder(nn.Module):
    """GMR or PbGMR encoder over a fixed graph."""

    def __init__(
        self,
        node_input_dim: int,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        reduction_indices: torch.Tensor,
        reduction_weights: torch.Tensor,
        *,
        width: int,
        latent_per_pivot: int,
        processor_blocks: int,
        residual_layernorm: bool,
    ) -> None:
        super().__init__()
        self.node_encoder = MLP(node_input_dim, width, width, hidden_layers=2)
        self.edge_encoder = MLP(
            edge_features.shape[-1], width, width, hidden_layers=2
        )
        self.processors = nn.ModuleList(
            [
                GraphNetBlock(width, residual_layernorm=residual_layernorm)
                for _ in range(processor_blocks)
            ]
        )
        self.readout = MLP(width, latent_per_pivot, width, hidden_layers=2)
        self.register_buffer("edge_features", edge_features, persistent=False)
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer(
            "reduction_indices", reduction_indices.long(), persistent=False
        )
        self.register_buffer(
            "reduction_weights", reduction_weights.float(), persistent=False
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        nodes = self.node_encoder(node_features)
        edge_seed = self.edge_encoder(self.edge_features)
        edges = edge_seed.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        for processor in self.processors:
            nodes, edges = processor(nodes, edges, self.edge_index)

        nodal_latents = self.readout(nodes)
        selected = nodal_latents[:, self.reduction_indices]
        pivotal = torch.sum(
            selected * self.reduction_weights[None, :, :, None], dim=2
        )
        return pivotal.reshape(pivotal.shape[0], -1)


class GraphMeshDecoder(nn.Module):
    """GMUS or PbGMUS decoder over a fixed graph."""

    def __init__(
        self,
        output_dim: int,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        expansion_indices: torch.Tensor,
        expansion_weights: torch.Tensor,
        *,
        pivotal_count: int,
        width: int,
        latent_per_pivot: int,
        processor_blocks: int,
        residual_layernorm: bool,
    ) -> None:
        super().__init__()
        self.latent_per_pivot = latent_per_pivot
        self.pivotal_count = pivotal_count
        self.node_encoder = MLP(
            latent_per_pivot, width, width, hidden_layers=2
        )
        self.edge_encoder = MLP(
            edge_features.shape[-1], width, width, hidden_layers=2
        )
        self.processors = nn.ModuleList(
            [
                GraphNetBlock(width, residual_layernorm=residual_layernorm)
                for _ in range(processor_blocks)
            ]
        )
        self.output = MLP(width, output_dim, width, hidden_layers=2)
        self.register_buffer("edge_features", edge_features, persistent=False)
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer(
            "expansion_indices", expansion_indices.long(), persistent=False
        )
        self.register_buffer(
            "expansion_weights", expansion_weights.float(), persistent=False
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        pivotal = latent.reshape(
            latent.shape[0], self.pivotal_count, self.latent_per_pivot
        )
        selected = pivotal[:, self.expansion_indices]
        nodes = torch.sum(
            selected * self.expansion_weights[None, :, :, None], dim=2
        )
        nodes = self.node_encoder(nodes)
        edge_seed = self.edge_encoder(self.edge_features)
        edges = edge_seed.unsqueeze(0).expand(latent.shape[0], -1, -1)
        for processor in self.processors:
            nodes, edges = processor(nodes, edges, self.edge_index)
        return self.output(nodes)


class MeshReducedAutoencoder(nn.Module):
    """GMR-GMUS or PbGMR-PbGMUS with the paper architecture."""

    def __init__(
        self,
        node_input_dim: int,
        output_dim: int,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        reduction_indices: torch.Tensor,
        reduction_weights: torch.Tensor,
        expansion_indices: torch.Tensor,
        expansion_weights: torch.Tensor,
        *,
        pivotal_count: int,
        width: int = 128,
        latent_per_pivot: int = 4,
        processor_blocks: int = 3,
        residual_layernorm: bool = False,
    ) -> None:
        super().__init__()
        if reduction_indices.shape[0] != pivotal_count:
            raise ValueError("reduction geometry does not match pivotal_count")
        if expansion_indices.max().item() >= pivotal_count:
            raise ValueError("expansion geometry references an invalid pivot")
        self.pivotal_count = pivotal_count
        self.latent_per_pivot = latent_per_pivot
        self.latent_dim = pivotal_count * latent_per_pivot
        self.encoder = GraphMeshEncoder(
            node_input_dim,
            edge_features,
            edge_index,
            reduction_indices,
            reduction_weights,
            width=width,
            latent_per_pivot=latent_per_pivot,
            processor_blocks=processor_blocks,
            residual_layernorm=residual_layernorm,
        )
        self.decoder = GraphMeshDecoder(
            output_dim,
            edge_features,
            edge_index,
            expansion_indices,
            expansion_weights,
            pivotal_count=pivotal_count,
            width=width,
            latent_per_pivot=latent_per_pivot,
            processor_blocks=processor_blocks,
            residual_layernorm=residual_layernorm,
        )

    def encode(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(node_features)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(node_features))


class GMRResidualTemporalTransformer(nn.Module):
    """Han et al.'s one-layer, four-head residual temporal attention model."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        *,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by attention heads")
        self.latent_dim = latent_dim
        self.heads = heads
        self.head_dim = latent_dim // heads
        self.condition_encoder = MLP(
            condition_dim, latent_dim, 100, hidden_layers=2
        )
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        # Equation (11): concatenated heads pass through an MLP that predicts
        # the residual over the previous latent.
        self.residual_mlp = MLP(
            latent_dim,
            latent_dim,
            4 * latent_dim,
            hidden_layers=1,
            zero_last=True,
        )

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(
            x.shape[0], x.shape[1], self.heads, self.head_dim
        ).transpose(1, 2)

    def rollout(
        self, initial: torch.Tensor, conditions: torch.Tensor
    ) -> torch.Tensor:
        if conditions.ndim != 3:
            raise ValueError("conditions must have shape [batch, horizon, dim]")
        if initial.shape[0] != conditions.shape[0]:
            raise ValueError("initial and conditions batch sizes differ")

        current = initial
        history_keys = [self._heads(self.key(initial[:, None]))]
        history_values = [self._heads(self.value(initial[:, None]))]
        predictions = []
        scale = math.sqrt(self.head_dim)

        for step in range(conditions.shape[1]):
            parameter = self.condition_encoder(conditions[:, step])[:, None]
            parameter_key = self._heads(self.key(parameter))
            parameter_value = self._heads(self.value(parameter))
            keys = torch.cat((parameter_key, *history_keys), dim=2)
            values = torch.cat((parameter_value, *history_values), dim=2)
            query = self._heads(self.query(current[:, None]))
            attention = torch.softmax(
                torch.matmul(query, keys.transpose(-2, -1)) / scale, dim=-1
            )
            context = torch.matmul(attention, values).transpose(1, 2)
            context = context.reshape(context.shape[0], self.latent_dim)
            current = current + self.residual_mlp(context)
            predictions.append(current)
            history_keys.append(self._heads(self.key(current[:, None])))
            history_values.append(self._heads(self.value(current[:, None])))
        return torch.stack(predictions, dim=1)

    def forward(
        self, initial: torch.Tensor, conditions: torch.Tensor
    ) -> torch.Tensor:
        return self.rollout(initial, conditions)


class ProjectedAttention(nn.Module):
    """Multi-head attention exposing projections for cached decoding."""

    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("attention dimension must divide evenly")
        self.dimension = dimension
        self.heads = heads
        self.head_dim = dimension // heads
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(
            x.shape[0], x.shape[1], self.heads, self.head_dim
        ).transpose(1, 2)

    def project_key(self, x: torch.Tensor) -> torch.Tensor:
        return self._heads(self.key(x))

    def project_value(self, x: torch.Tensor) -> torch.Tensor:
        return self._heads(self.value(x))

    def attend_projected(
        self,
        query_input: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self._heads(self.query(query_input))
        scores = torch.matmul(query, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask[None, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, values).transpose(1, 2)
        context = context.reshape(
            context.shape[0], context.shape[1], self.dimension
        )
        return self.output(context)

    def forward(
        self,
        query_input: torch.Tensor,
        key_input: torch.Tensor,
        value_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.attend_projected(
            query_input,
            self.project_key(key_input),
            self.project_value(value_input),
            mask,
        )


class PbTransformerConditioner(nn.Module):
    """Two-layer encoder and one-layer masked decoder from Sun et al."""

    def __init__(
        self, latent_dim: int, condition_dim: int, *, heads: int = 4
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_encoder = MLP(
            condition_dim, latent_dim, 100, hidden_layers=2
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=4 * latent_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.source_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False
        )
        self.self_attention = ProjectedAttention(latent_dim, heads)
        self.self_norm = nn.LayerNorm(latent_dim)
        self.cross_attention = ProjectedAttention(latent_dim, heads)
        self.cross_norm = nn.LayerNorm(latent_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def encode_source(
        self, initial: torch.Tensor, initial_condition: torch.Tensor
    ) -> torch.Tensor:
        parameter = self.condition_encoder(initial_condition)
        source = torch.stack((parameter, initial), dim=1)
        return self.source_encoder(source)

    def teacher_context(
        self,
        latent_sequence: torch.Tensor,
        conditions: torch.Tensor,
        *,
        temporal_conditioning: bool,
    ) -> torch.Tensor:
        horizon = latent_sequence.shape[1] - 1
        if conditions.shape[1] != horizon:
            raise ValueError("condition horizon does not match latent sequence")
        memory = self.encode_source(latent_sequence[:, 0], conditions[:, 0])
        decoder_input = latent_sequence[:, :-1]
        if temporal_conditioning:
            decoder_input = decoder_input + self.condition_encoder(conditions)
        causal_mask = torch.triu(
            torch.ones(
                horizon,
                horizon,
                dtype=torch.bool,
                device=latent_sequence.device,
            ),
            diagonal=1,
        )
        attended = self.self_attention(
            decoder_input, decoder_input, decoder_input, causal_mask
        )
        hidden = self.self_norm(decoder_input + attended)
        crossed = self.cross_attention(hidden, memory, memory)
        hidden = self.cross_norm(hidden + crossed)
        return self.output_norm(hidden + self.feed_forward(hidden))

    def prepare_generation(
        self, initial: torch.Tensor, initial_condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encode_source(initial, initial_condition)
        return (
            self.cross_attention.project_key(memory),
            self.cross_attention.project_value(memory),
        )

    def generation_step(
        self,
        raw_input: torch.Tensor,
        self_keys: torch.Tensor,
        self_values: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
    ) -> torch.Tensor:
        raw_input = raw_input[:, None]
        attended = self.self_attention.attend_projected(
            raw_input, self_keys, self_values
        )
        hidden = self.self_norm(raw_input + attended)
        crossed = self.cross_attention.attend_projected(
            hidden, memory_keys, memory_values
        )
        hidden = self.cross_norm(hidden + crossed)
        hidden = self.output_norm(hidden + self.feed_forward(hidden))
        return hidden[:, 0]


class ConditionalCouplingNetwork(nn.Module):
    """Scale/translation network from the authors' released RealNVP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        *,
        hidden_layers: int,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(hidden_layers):
            layers.extend((activation(), nn.Linear(hidden_dim, hidden_dim)))
        layers.extend((activation(), nn.Linear(hidden_dim, output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class ConditionalMaskedCoupling(nn.Module):
    """Author-source alternating-mask conditional affine coupling."""

    def __init__(
        self,
        dimension: int,
        condition_dim: int,
        hidden_dim: int,
        *,
        mask: torch.Tensor,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask.float())
        network_input_dim = dimension + condition_dim
        self.scale = ConditionalCouplingNetwork(
            network_input_dim,
            dimension,
            hidden_dim,
            hidden_layers=hidden_layers,
            activation=nn.Tanh,
        )
        self.shift = ConditionalCouplingNetwork(
            network_input_dim,
            dimension,
            hidden_dim,
            hidden_layers=hidden_layers,
            activation=nn.ReLU,
        )

    def _coupling_parameters(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = value * self.mask
        # The released implementation concatenates condition before the full
        # masked vector, uses tanh scale subnetworks and ReLU shift subnetworks,
        # and bounds log scale with one final tanh.
        inputs = torch.cat((condition, masked), dim=-1)
        transformed_mask = 1.0 - self.mask
        log_scale = torch.tanh(self.scale(inputs)) * transformed_mask
        shift = self.shift(inputs) * transformed_mask
        return log_scale, shift

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = self._coupling_parameters(value, condition)
        transformed = value * torch.exp(log_scale) + shift
        return transformed, log_scale.sum(dim=-1)

    def inverse(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = self._coupling_parameters(value, condition)
        transformed = (value - shift) * torch.exp(-log_scale)
        return transformed, -log_scale.sum(dim=-1)


class FlowBatchNorm(nn.Module):
    """Invertible RealNVP batch normalization from the author release."""

    def __init__(
        self, dimension: int, momentum: float = 0.9, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.log_gamma = nn.Parameter(torch.zeros(dimension))
        self.beta = nn.Parameter(torch.zeros(dimension))
        self.register_buffer("running_mean", torch.zeros(dimension))
        self.register_buffer("running_var", torch.ones(dimension))
        self._batch_mean: Optional[torch.Tensor] = None
        self._batch_var: Optional[torch.Tensor] = None

    def _statistics(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            flattened = value.reshape(-1, value.shape[-1])
            mean = flattened.mean(dim=0)
            # The source comment requests the biased estimate even though its
            # original torch.var call omitted the flag.
            variance = flattened.var(dim=0, unbiased=False)
            self._batch_mean = mean
            self._batch_var = variance
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(
                    mean.detach() * (1.0 - self.momentum)
                )
                self.running_var.mul_(self.momentum).add_(
                    variance.detach() * (1.0 - self.momentum)
                )
            return mean, variance
        return self.running_mean, self.running_var

    def forward(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, variance = self._statistics(value)
        normalized = (value - mean) / torch.sqrt(variance + self.eps)
        transformed = torch.exp(self.log_gamma) * normalized + self.beta
        logdet = (
            self.log_gamma - 0.5 * torch.log(variance + self.eps)
        ).sum()
        return transformed, logdet.expand(value.shape[0])

    def inverse(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            if self._batch_mean is None or self._batch_var is None:
                raise RuntimeError(
                    "training-mode flow batch norm has no batch statistics"
                )
            mean, variance = self._batch_mean, self._batch_var
        else:
            mean, variance = self.running_mean, self.running_var
        normalized = (value - self.beta) * torch.exp(-self.log_gamma)
        transformed = normalized * torch.sqrt(variance + self.eps) + mean
        logdet = (
            0.5 * torch.log(variance + self.eps) - self.log_gamma
        ).sum()
        return transformed, logdet.expand(value.shape[0])


class ConditionalRealNVP(nn.Module):
    """Sun et al. RealNVP aligned with the pinned author source."""

    def __init__(
        self,
        dimension: int,
        condition_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        coupling_layers: int = 2,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dimension
        self.dimension = dimension
        initial_mask = torch.arange(dimension).float() % 2
        self.couplings = nn.ModuleList(
            [
                ConditionalMaskedCoupling(
                    dimension,
                    condition_dim,
                    hidden_dim,
                    mask=(
                        initial_mask
                        if index % 2 == 0
                        else 1.0 - initial_mask
                    ),
                    hidden_layers=2,
                )
                for index in range(coupling_layers)
            ]
        )
        self.normalizations = nn.ModuleList(
            [FlowBatchNorm(dimension) for _ in range(coupling_layers)]
        )

    def sample(
        self, base: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        value = base
        for coupling, normalization in reversed(
            list(zip(self.couplings, self.normalizations))
        ):
            value, _ = normalization.inverse(value)
            value, _ = coupling.inverse(value, condition)
        return value

    def log_prob(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        base = value
        forward_logdet = torch.zeros(
            value.shape[0], dtype=value.dtype, device=value.device
        )
        for coupling, normalization in zip(
            self.couplings, self.normalizations
        ):
            base, coupling_logdet = coupling(base, condition)
            base, normalization_logdet = normalization(base)
            forward_logdet = (
                forward_logdet + coupling_logdet + normalization_logdet
            )
        base_log_prob = -0.5 * (
            torch.square(base) + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        return base_log_prob + forward_logdet


class PbGMRRealNVPTemporal(nn.Module):
    """Sun et al. conditional Transformer/RealNVP temporal head."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        *,
        heads: int = 4,
        coupling_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.conditioner = PbTransformerConditioner(
            latent_dim, condition_dim, heads=heads
        )
        self.flow = ConditionalRealNVP(
            latent_dim,
            latent_dim,
            hidden_dim=latent_dim,
            coupling_layers=coupling_layers,
        )

    def negative_log_likelihood(
        self,
        latent_sequence: torch.Tensor,
        conditions: torch.Tensor,
        *,
        temporal_conditioning: bool,
    ) -> torch.Tensor:
        contexts = self.conditioner.teacher_context(
            latent_sequence,
            conditions,
            temporal_conditioning=temporal_conditioning,
        )
        targets = latent_sequence[:, 1:]
        log_probability = self.flow.log_prob(
            targets.reshape(-1, self.latent_dim),
            contexts.reshape(-1, self.latent_dim),
        )
        # Per-dimension NLL keeps validation magnitudes comparable across
        # datasets whose pivotal counts differ.
        return -log_probability.mean() / self.latent_dim

    def rollout(
        self,
        initial: torch.Tensor,
        conditions: torch.Tensor,
        *,
        sample: bool,
        generator: Optional[torch.Generator] = None,
        temporal_conditioning: bool,
    ) -> torch.Tensor:
        memory_keys, memory_values = self.conditioner.prepare_generation(
            initial, conditions[:, 0]
        )
        self_keys: Optional[torch.Tensor] = None
        self_values: Optional[torch.Tensor] = None
        current = initial
        predictions = []

        for step in range(conditions.shape[1]):
            raw = current
            if temporal_conditioning:
                raw = raw + self.conditioner.condition_encoder(
                    conditions[:, step]
                )
            raw_token = raw[:, None]
            next_key = self.conditioner.self_attention.project_key(raw_token)
            next_value = self.conditioner.self_attention.project_value(raw_token)
            self_keys = (
                next_key
                if self_keys is None
                else torch.cat((self_keys, next_key), dim=2)
            )
            self_values = (
                next_value
                if self_values is None
                else torch.cat((self_values, next_value), dim=2)
            )
            context = self.conditioner.generation_step(
                raw, self_keys, self_values, memory_keys, memory_values
            )
            if sample:
                base = torch.randn(
                    current.shape,
                    dtype=current.dtype,
                    device=current.device,
                    generator=generator,
                )
            else:
                base = torch.zeros_like(current)
            current = self.flow.sample(base, context)
            predictions.append(current)
        return torch.stack(predictions, dim=1)

    def forward(
        self,
        latent_sequence: torch.Tensor,
        conditions: torch.Tensor,
        *,
        temporal_conditioning: bool,
    ) -> torch.Tensor:
        return self.negative_log_likelihood(
            latent_sequence,
            conditions,
            temporal_conditioning=temporal_conditioning,
        )
