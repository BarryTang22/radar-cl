"""PGSU Adapter Modules (Paper Sections III-B, III-C).

Provides LoRA adapters for transformer models and bottleneck adapters for CNN models,
with novelty-controlled rank masking and width gating per the PGSU paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_first(k_t, k_max):
    """MASKFIRST(k_t, k_max) — binary mask with first k_t entries = 1 (Alg. 1, line 10).

    Args:
        k_t: Number of active entries (int)
        k_max: Total mask length (int)

    Returns:
        Tensor of shape (k_max,) with first k_t entries = 1, rest = 0.
    """
    mask = torch.zeros(k_max)
    mask[:k_t] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Transformer path: LoRA adapters (Eq. 13-15)
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """Single LoRA adapter for one nn.Linear projection (Eq. 13-15).

    Computes additive delta: delta = B(mask * A(x)).
    lora_A projects input to low-rank space, lora_B projects back.
    rank_mask controls the effective rank via MASKFIRST.
    """

    def __init__(self, in_features, out_features, k_max):
        super().__init__()
        self.k_max = k_max
        self.lora_A = nn.Linear(in_features, k_max, bias=False)
        self.lora_B = nn.Linear(k_max, out_features, bias=False)
        self.register_buffer('rank_mask', torch.ones(k_max))

        # Kaiming init for A, zero init for B (standard LoRA practice)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def set_rank_mask(self, mask):
        """Update the active rank mask."""
        self.rank_mask.copy_(mask)

    def forward(self, x):
        """Compute LoRA delta: B(mask * A(x))."""
        h = self.lora_A(x)
        h = h * self.rank_mask
        return self.lora_B(h)


class LoRAInjector(nn.Module):
    """Hook-based LoRA injection for RadarTransformer (Eq. 13-15).

    Injects LoRA into temporal_transformer.layers[i].self_attn.{q_proj, v_proj}
    using PyTorch forward hooks. Does NOT modify the base model weights.
    """

    def __init__(self, model, k_max, target_modules=('q_proj', 'v_proj')):
        super().__init__()
        self.k_max = k_max
        self.target_modules = target_modules
        self._hooks = []

        # Create LoRA layers for each target in each transformer layer
        self.lora_layers = nn.ModuleDict()
        for layer_idx, layer in enumerate(model.temporal_transformer.layers):
            attn = layer.self_attn
            for name in target_modules:
                linear = getattr(attn, name)
                key = f'layer{layer_idx}_{name}'
                self.lora_layers[key] = LoRALayer(
                    linear.in_features, linear.out_features, k_max
                )

        # Store model ref without making it a submodule (avoids double-counting params)
        object.__setattr__(self, '_model', model)

    def install_hooks(self):
        """Register forward hooks on target linear layers."""
        self.remove_hooks()
        for layer_idx, layer in enumerate(self._model.temporal_transformer.layers):
            attn = layer.self_attn
            for name in self.target_modules:
                linear = getattr(attn, name)
                key = f'layer{layer_idx}_{name}'
                lora = self.lora_layers[key]

                def make_hook(lora_layer):
                    def hook(module, input, output):
                        return output + lora_layer(input[0])
                    return hook

                handle = linear.register_forward_hook(make_hook(lora))
                self._hooks.append(handle)

    def remove_hooks(self):
        """Remove all installed hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def set_rank_mask(self, mask):
        """Broadcast mask to all LoRA layers."""
        for lora in self.lora_layers.values():
            lora.set_rank_mask(mask)

    def forward(self, x):
        """Not used directly — hooks handle injection."""
        return x


# ---------------------------------------------------------------------------
# CNN path: bottleneck adapters + feature prompt (Eq. 26-31)
# ---------------------------------------------------------------------------

class CNNQueryModule(nn.Module):
    """Query extraction for CNN models (Eq. 27): q = W_cnn * Pool(features)."""

    def __init__(self, feature_dim=512, query_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, query_dim)

    def forward(self, features):
        """Project pooled features to query space."""
        return self.query_proj(features)


class CNNFeaturePrompt(nn.Module):
    """Query-conditioned feature prompt (Eq. 26): p^cnn_t(x) = P_phi(q(x)).

    MLP that generates a feature-space prompt vector from the query.
    Output layer is zero-initialized for stable residual behavior.
    """

    def __init__(self, query_dim=128, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        # Zero-init output layer for stable residual
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, query):
        """Generate feature-space prompt from query."""
        return self.mlp(query)


class CNNBottleneckAdapter(nn.Module):
    """Bottleneck adapter (Eq. 26): h' = h_tilde + alpha_t * B(sigma(mask * A(h_tilde))).

    Width-masked bottleneck with external gate alpha_t.
    """

    def __init__(self, feature_dim=512, r_adp_max=64):
        super().__init__()
        self.r_adp_max = r_adp_max
        self.adapter_down = nn.Linear(feature_dim, r_adp_max, bias=False)
        self.adapter_up = nn.Linear(r_adp_max, feature_dim, bias=False)
        self.register_buffer('width_mask', torch.ones(r_adp_max))
        self.alpha_t = 1.0  # Set externally by PGSU controller

        # Kaiming init for down, zero init for up (LoRA-style)
        nn.init.kaiming_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)

    def set_width_mask(self, mask):
        """Update the active adapter width mask."""
        self.width_mask.copy_(mask)

    def forward(self, h):
        """Apply bottleneck adapter with width mask and gate."""
        down = self.adapter_down(h)
        down = F.relu(down * self.width_mask)
        delta = self.adapter_up(down)
        return h + self.alpha_t * delta


class CNNPGSUWrapper(nn.Module):
    """CNN forward wrapper with PGSU modules (Eq. 26-31).

    Wraps ResNet18 to inject query, feature prompt, and bottleneck adapter
    without modifying the base model.
    """

    def __init__(self, model, feature_dim=512, query_dim=128, r_adp_max=64):
        super().__init__()
        # Store model ref without making it a submodule (avoids double-counting params)
        object.__setattr__(self, 'model', model)
        self.query_module = CNNQueryModule(feature_dim, query_dim)
        self.feature_prompt = CNNFeaturePrompt(query_dim, feature_dim)
        self.adapter = CNNBottleneckAdapter(feature_dim, r_adp_max)
        self._last_prompt_output = None

    def get_query(self, x):
        """Extract query from backbone features (detached)."""
        with torch.no_grad():
            features = self.model.get_features(x)
        return self.query_module(features)

    def forward(self, x):
        """Full forward: features + prompt + adapter + dropout + classifier."""
        features = self.model.get_features(x)
        query = self.query_module(features.detach())
        prompt = self.feature_prompt(query)
        self._last_prompt_output = prompt
        h = features + prompt
        h = self.adapter(h)
        h = self.model.fc_dropout(h)
        return self.model.classifier(h)

    def prompt_loss(self):
        """L2 norm on feature prompt output (L_prompt for CNN branch)."""
        if self._last_prompt_output is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._last_prompt_output.norm(dim=-1).mean()
