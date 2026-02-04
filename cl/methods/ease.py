"""EASE - Expandable Subspace Ensemble."""
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import CosineLinear, EASEAdapter


class EASE(nn.Module):
    """EASE - Expandable Subspace Ensemble for Continual Learning.

    Paper: Zhou et al. "Expandable Subspace Ensemble for Pre-Trained Model-Based
           Class-Incremental Learning" (CVPR 2024)
    Official: https://github.com/sun-hailong/CVPR24-Ease

    Key features:
    - CONCATENATES features from all adapters (not sequential)
    - Uses CosineLinear classifier (prototype-based)
    - Feature reweighting with alpha for previous tasks
    - Growing feature dimension: embed_dim * num_tasks

    Args:
        embed_dim: Backbone embedding dimension
        bottleneck_dim: Adapter bottleneck dimension
        alpha: Reweight factor for previous task features
        use_init_ptm: Include original backbone features
        beta: Scale for backbone features if use_init_ptm
    """

    def __init__(self, embed_dim=128, bottleneck_dim=16, alpha=0.1, use_init_ptm=False, beta=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.alpha = alpha
        self.use_init_ptm = use_init_ptm
        self.beta = beta
        self.adapters = nn.ModuleList()
        self.classifier = None
        self.task_id = -1
        self.num_tasks = 0
        self.prototypes = {}

    def to(self, device):
        """Move module to device."""
        return super().to(device)

    def get_output_dim(self):
        """Feature dimension grows with tasks."""
        base = self.embed_dim if self.use_init_ptm else 0
        return base + self.num_tasks * self.embed_dim

    def add_task(self):
        """Add new adapter for new task and freeze old ones."""
        self.task_id += 1
        self.num_tasks += 1
        adapter = EASEAdapter(self.embed_dim, self.bottleneck_dim)
        self.adapters.append(adapter)
        for i, adapt in enumerate(self.adapters[:-1]):
            for p in adapt.parameters():
                p.requires_grad = False

    def create_or_expand_classifier(self, num_classes, device):
        """Create or expand CosineLinear classifier for new classes."""
        output_dim = self.get_output_dim()
        if self.classifier is None:
            self.classifier = CosineLinear(output_dim, num_classes).to(device)
        else:
            if num_classes > self.classifier.num_classes:
                self.classifier.expand_classes(num_classes, device)
            if output_dim > self.classifier.in_features:
                old_weight = self.classifier.weight.data
                new_weight = torch.zeros(self.classifier.num_classes, output_dim, device=device)
                new_weight[:, :old_weight.size(1)] = old_weight
                self.classifier.weight = nn.Parameter(new_weight)
                self.classifier.in_features = output_dim

    def parameters(self):
        """Return trainable parameters (current adapter + classifier)."""
        params = []
        if self.task_id >= 0 and self.task_id < len(self.adapters):
            params.extend(self.adapters[self.task_id].parameters())
        if self.classifier is not None:
            params.extend(self.classifier.parameters())
        return params

    def forward_features(self, x, training=True):
        """CONCATENATE features from all adapters."""
        features_list = []

        if self.use_init_ptm:
            features_list.append(self.beta * x)

        for i, adapter in enumerate(self.adapters):
            adapter_feat = adapter(x)
            if i < self.task_id:
                adapter_feat = self.alpha * adapter_feat
            features_list.append(adapter_feat)

        if not features_list:
            return x

        return torch.cat(features_list, dim=1)

    def forward(self, x, training=True):
        """Full forward pass: features -> classifier."""
        features = self.forward_features(x, training)
        if self.classifier is None:
            raise RuntimeError("Classifier not initialized. Call create_or_expand_classifier first.")
        return self.classifier(features)

    def extract_prototypes(self, dataloader, model, device, task_classes):
        """Extract class mean features as prototypes and update old class prototypes."""
        self.eval()
        model.eval()

        class_features = defaultdict(list)

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(device)
                backbone_features = model.get_features(data)
                concat_features = self.forward_features(backbone_features, training=False)

                for i, label in enumerate(labels):
                    class_features[label.item()].append(concat_features[i].cpu())

        for class_id, feats in class_features.items():
            if feats:
                prototype = torch.stack(feats).mean(dim=0)
                self.prototypes[class_id] = prototype

                if self.classifier is not None and class_id < self.classifier.num_classes:
                    self.classifier.weight.data[class_id] = prototype.to(device)

        # Update old class prototypes in the new subspace using similarity-based reconstruction
        if self.task_id > 0:
            self._solve_similarity(task_classes, device)

    def _solve_similarity(self, new_task_classes, device):
        """Update old class prototypes in new subspace using cosine similarity weighting.

        For old classes (from previous tasks), we don't have data to extract features
        in the new adapter's subspace. Instead, we use cosine similarity between
        old and new class prototypes to reconstruct the missing dimensions.

        Reference: CVPR 2024 EASE paper - Semantic-guided Prototype Complement
        """
        if self.classifier is None or self.task_id < 1:
            return

        new_task_classes = set(new_task_classes) if not isinstance(new_task_classes, set) else new_task_classes
        old_classes = [c for c in self.prototypes.keys() if c not in new_task_classes]

        if not old_classes or not new_task_classes:
            return

        # Get the current output dimension and new adapter's feature range
        total_dim = self.get_output_dim()
        base_offset = self.embed_dim if self.use_init_ptm else 0
        new_adapter_start = base_offset + self.task_id * self.embed_dim
        new_adapter_end = new_adapter_start + self.embed_dim

        # Extract new task prototypes (they have valid features in new subspace)
        new_prototypes = []
        new_class_list = sorted(new_task_classes)
        for c in new_class_list:
            if c in self.prototypes:
                new_prototypes.append(self.prototypes[c])
        if not new_prototypes:
            return
        new_proto_tensor = torch.stack(new_prototypes).to(device)  # (num_new, total_dim)

        # For each old class, compute similarity-weighted combination of new class features
        for old_class in old_classes:
            if old_class not in self.prototypes:
                continue
            old_proto = self.prototypes[old_class].to(device)  # (old_dim,)

            # Expand old prototype to new dimension if needed
            if old_proto.size(0) < total_dim:
                expanded_proto = torch.zeros(total_dim, device=device)
                expanded_proto[:old_proto.size(0)] = old_proto
                old_proto = expanded_proto

            # Use features from shared subspaces (before new adapter) for similarity
            shared_dim = new_adapter_start
            old_shared = old_proto[:shared_dim]
            new_shared = new_proto_tensor[:, :shared_dim]

            # Compute cosine similarity
            old_norm = F.normalize(old_shared.unsqueeze(0), p=2, dim=1)  # (1, shared_dim)
            new_norm = F.normalize(new_shared, p=2, dim=1)  # (num_new, shared_dim)
            similarity = torch.mm(old_norm, new_norm.t()).squeeze(0)  # (num_new,)

            # Softmax to get weights
            weights = F.softmax(similarity, dim=0)  # (num_new,)

            # Weighted combination of new adapter features
            new_adapter_features = new_proto_tensor[:, new_adapter_start:new_adapter_end]  # (num_new, embed_dim)
            reconstructed = torch.mm(weights.unsqueeze(0), new_adapter_features).squeeze(0)  # (embed_dim,)

            # Update the old class prototype with reconstructed features in new subspace
            old_proto[new_adapter_start:new_adapter_end] = reconstructed
            self.prototypes[old_class] = old_proto.cpu()

            # Update classifier weight
            if old_class < self.classifier.num_classes:
                self.classifier.weight.data[old_class] = old_proto

    def train(self, mode=True):
        """Set training mode."""
        if self.task_id >= 0 and self.task_id < len(self.adapters):
            self.adapters[self.task_id].train(mode)
        if self.classifier is not None:
            self.classifier.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        for adapter in self.adapters:
            adapter.eval()
        if self.classifier is not None:
            self.classifier.eval()
        return self
