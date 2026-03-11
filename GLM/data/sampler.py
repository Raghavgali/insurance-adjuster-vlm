from __future__ import annotations

import math
import random 
from typing import Iterator, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    """
    Batch sampler that groups samples with similar sequence lengths to reduce padding waste.

    Parameters
    ----------
    lengths: Sequence[int]
        Per-sample token lengths (one length per dataset index).
    batch_size: int
        Number of samples per batch.
    bucket_size: int, default=256
        Number of sorted indices per bucket before forming batches.
        Larger values improve length homogeneity but reduce randomness.
    shuffle: bool, default=True
        Whether to shuffle indices within buckets and shuffle final batch order.
    drop_last: bool, default=False
        Whether to drop the last incomplete batch.
    seed: int, default=42
        Base random seed. Use set_epoch(epoch) each epoch for deterministic reshuffling.
    num_replicas: int | None, default=None
        Number of DDP processes. If None, inferred from torch.distributed when initialized.
    rank: int | None, default=None
        Current DDP rank. If None, inferred from torch.distributed when initialized.

    Returns
    -------
    LengthBucketBatchSampler
        A sampler compatible with DataLoader(batch_sampler=...).

    Notes
    -----
    - Works with both single-process and DDP.
    - In DDP, batches are sharded by rank (every Nth global batch).
    - Do not pass batch_size/shuffle/sampler to DataLoader when using batch_sampler.
    """
    def __init__(self,
                 lengths: Sequence[int],
                 batch_size: int,
                 *,
                 bucket_size: int = 256,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 seed: int = 42,
                 num_replicas: int | None = None,
                 rank: int | None = None,
                 ) -> None: 
        if not isinstance(lengths, Sequence) or len(lengths) == 0:
            raise ValueError("`lengths` must be a non-empty sequence of positive integers")
        if any((not isinstance(x, int)) or x <= 0 for x in lengths):
            raise ValueError("All values in `lengths` must be positive integers")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer")
        if not isinstance(bucket_size, int) or bucket_size <= 0:
            raise ValueError("`bucket_size` must be a positive integer")
        if not isinstance(shuffle, bool):
            raise TypeError("`shuffle` must be bool")
        if not isinstance(drop_last, bool):
            raise TypeError("`drop_last` must be bool")
        if not isinstance(seed, int):
            raise TypeError("`seed` must be int")

        if num_replicas is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                inferred_world = dist.get_world_size()
                inferred_rank = dist.get_rank()
                num_replicas = inferred_world if num_replicas is None else num_replicas
                rank = inferred_rank if rank is None else rank
            else:
                num_replicas = 1 if num_replicas is None else num_replicas
                rank = 0 if rank is None else rank 

        
        if not isinstance(num_replicas, int) or num_replicas <= 0:
            raise ValueError("`num_replicas` must be a positive integer")
        if not isinstance(rank, int) or not (0 <= rank < num_replicas):
            raise ValueError("`rank` must satisfy 0 <= rank < num_replicas")

        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.num_replicas = num_replicas
        self.rank = rank

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic reshuffling across epochs."""
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("`epoch` must be non-negative integer")
        self.epoch = epoch 

    def _build_global_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)

        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        indices.sort(key=lambda i: self.lengths[i])

        buckets = [
            indices[i : i + self.bucket_size]
            for i in range(0, len(indices), self.bucket_size)
        ]

        batches: list[list[int]] = []
        for bucket in buckets:
            if self.shuffle:
                rng.shuffle(bucket)

            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        return batches
    
    def __iter__(self):
        global_batches = self._build_global_batches()

        for batch_idx, batch in enumerate(global_batches):
            if batch_idx % self.num_replicas == self.rank:
                yield batch
        
    def __len__(self):
        total_samples = len(self.lengths)
        if self.drop_last:
            total_global_batches = total_samples // self.batch_size
        else:
            total_global_batches = math.ceil(total_samples / self.batch_size)

        # Per-rank batch count after strided sharding of global batches.
        return math.ceil(total_global_batches / self.num_replicas)
    
    