# Baseline rerun provenance

Only results produced by the common held-out evaluator in this directory are
eligible for the directly comparable table. Published single-trajectory values
remain separate.

## Shared protocol

- Trajectory split seed: 42.
- Cylinder: 61 train / 20 validation / 20 test, 400 predicted steps.
- Multiphase: 24 train / 8 validation / 8 test, 199 predicted steps.
- WE1 and E1: 2048 train / 129 validation / 127 test, 248 predicted steps.
- Splits are byte-for-byte identical to the paper evaluation manifest for
  cylinder and multiphase.
- Normalization statistics use training trajectories only.
- Selection uses validation data only.
- GMR/PbGMR spatial encoders receive physical state fields only, matching the
  papers; physical parameters enter their temporal models as condition tokens.
- For each `(trajectory, forecast step, field)`, spatial nodes are reduced as
  `sum((prediction - target)^2) / sum(target^2)`. Values are then averaged
  uniformly over held-out trajectories and steps; the macro value is the
  arithmetic mean across fields.

## MGN and MGN-NI

- Original authoritative source:
  <https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets>
- Local source commit:
  `f5de0ede8430809180254ee957abf36ed62579ef`
- Executable maintained PyTorch/PyG implementation:
  <https://github.com/NVIDIA/physicsnemo>
- Local source commit:
  `77b3c68001159b948a16804fa76eb127735fb6d1`
- Configuration: 15 message-passing blocks, width 128, two-layer node/edge
  MLPs, sum aggregation, and delta-state prediction.
- MGN-NI is the same architecture with Gaussian training noise of standard
  deviation 0.02 in normalized state space. MGN uses no noise.
- The paper arrays do not contain DeepMind's categorical `node_type` channel.
  Reruns therefore use exactly the available dynamic fields plus the same
  conditioning features supplied to GraphSpectralFormer; geometry enters
  through normalized relative edge coordinates and edge length. No held-out
  boundary values are injected during rollout.

The original release requires TensorFlow 1 and Sonnet 1, which do not support
the workspace's Python 3.11 runtime. The PhysicsNeMo implementation preserves
the published MGN architecture in the available PyTorch/PyG runtime.

## GMR-GMUS Transformer

- Paper: Han et al., *Predicting Physics in Mesh-reduced Space with Temporal
  Attention*, ICLR 2022.
- No author-maintained standalone repository was found.
- Maintained open implementation:
  <https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/vortex_shedding_mesh_reduced>
- The rerun records any code reconstructed from the paper equations as a
  paper-faithful reimplementation, not as author-released code.

## PbGMR-GMUS Transformer-RealNVP

- Paper: Sun et al., *Unifying Predictions of Deterministic and Stochastic
  Physics in Mesh-reduced Space with Sequential Flow Generative Model*,
  NeurIPS 2023.
- Author temporal/flow repository:
  <https://github.com/luningsun/Unified_Sequential_Flow_Generative_Model>,
  pinned locally at commit
  `383882bb56b33c48127a1874384e9897b5cde6e4` under
  `upstream/Unified_Sequential_Flow_Generative_Model/`.
- The PbGMR/GMUS encoder-decoder is available in PhysicsNeMo as
  `physicsnemo.models.mesh_reduced.Mesh_Reduced`.
- The author repository contains conditional RealNVP and Transformer training
  over precomputed flattened latents, but not the graph autoencoder, raw data,
  or latent-generation pipeline. The rerun therefore records the integration
  between the official temporal component and PhysicsNeMo spatial component.

## ViT-SEA

- Official repository: <https://github.com/anonymous/SEA>
- Local source commit:
  `59dffe0c03510e695be9cc26bd637d0a741f6735`
- The local matched-split configs use isolated checkpoint directories and do
  not overwrite historical or proposed-model artifacts.
