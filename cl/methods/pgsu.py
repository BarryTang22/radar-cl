"""Prompt-Guided Selective Update (PGSU)."""

import random
import torch
import torch.nn.functional as F


class PGSU:
    """Prompt-Guided Selective Update.

    Uses prompt similarity to decide which backbone parameters to update.
    High similarity to previous tasks → smaller backbone LR (reuse features)
    Low similarity (novel task) → larger backbone LR (learn new features)
    """

    def __init__(self, base_lr=1e-3, min_lr_scale=0.01, max_lr_scale=1.0,
                 min_replay=0.1, max_replay=0.5, buffer_size=300):
        self.base_lr = base_lr
        self.min_lr_scale = min_lr_scale
        self.max_lr_scale = max_lr_scale
        self.min_replay = min_replay
        self.max_replay = max_replay
        self.task_centroids = []
        self.buffer_size = buffer_size
        self.buffer = []
        self.current_novelty = 1.0

    def compute_task_novelty(self, query_centroid):
        """Compute how novel this task is compared to previous tasks."""
        if not self.task_centroids:
            self.current_novelty = 1.0
            return 1.0

        similarities = [F.cosine_similarity(query_centroid.unsqueeze(0), c.unsqueeze(0), dim=1).item()
                       for c in self.task_centroids]
        max_sim = max(similarities)
        novelty = 1.0 - max_sim
        self.current_novelty = max(0.0, min(1.0, novelty))
        return self.current_novelty

    def get_backbone_lr_scale(self, novelty):
        """Map novelty to learning rate scale."""
        return self.min_lr_scale + novelty * (self.max_lr_scale - self.min_lr_scale)

    def get_replay_ratio(self, novelty=None):
        """Get adaptive replay ratio based on task novelty (continuous).
        High novelty → min_replay (focus on new data)
        Low novelty → max_replay (replay more to prevent forgetting)
        """
        if novelty is None:
            novelty = self.current_novelty
        return self.max_replay - novelty * (self.max_replay - self.min_replay)

    def add_task_centroid(self, centroid):
        """Store centroid for the current task."""
        self.task_centroids.append(centroid.detach().clone())

    def add_samples(self, dataloader, num_samples=None):
        """Add samples to replay buffer."""
        samples = []
        for data, labels in dataloader:
            for i in range(data.size(0)):
                samples.append((data[i].cpu(), labels[i].item()))
        if num_samples is None:
            num_samples = min(len(samples), self.buffer_size // 3)
        selected = random.sample(samples, min(num_samples, len(samples)))
        self.buffer.extend(selected)
        if len(self.buffer) > self.buffer_size:
            self.buffer = random.sample(self.buffer, self.buffer_size)

    def get_batch(self, batch_size):
        """Get a batch from replay buffer."""
        if len(self.buffer) == 0:
            return None, None
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        data = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch])
        return data, labels


def compute_task_centroid(model, dataloader, device):
    """Compute mean query vector for a task's data."""
    was_training = model.training
    model.eval()
    all_queries = []

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            query = model.get_query(data)
            all_queries.append(query)

    all_queries = torch.cat(all_queries, dim=0)
    centroid = all_queries.mean(dim=0)
    if was_training:
        model.train()
    return centroid
