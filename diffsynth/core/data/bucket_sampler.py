import random
from collections import OrderedDict

import torch


class OrientationBucketSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, bucket_column="bucket", shuffle=True, seed=0):
        self.dataset = dataset
        self.bucket_column = bucket_column
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.bucket_to_indices = self._build_buckets()

    def _bucket_from_row(self, row, row_id):
        bucket = row.get(self.bucket_column)
        if bucket:
            return str(bucket)
        height = row.get("height")
        width = row.get("width")
        if height not in (None, "") and width not in (None, ""):
            return "landscape" if int(width) >= int(height) else "portrait"
        raise ValueError(f"Dataset row {row_id} is missing `{self.bucket_column}` metadata.")

    def _build_buckets(self):
        if not hasattr(self.dataset, "data"):
            raise ValueError("OrientationBucketSampler requires a UnifiedDataset-like object with `.data`.")
        base_length = len(self.dataset.data)
        dataset_length = len(self.dataset)
        if base_length <= 0 or dataset_length <= 0:
            raise ValueError("No bucket metadata found in dataset.")
        bucket_to_indices = OrderedDict()
        for dataset_index in range(dataset_length):
            row_id = dataset_index % base_length
            row = self.dataset.data[row_id]
            bucket = self._bucket_from_row(row, row_id)
            bucket_to_indices.setdefault(bucket, []).append(dataset_index)
        if not bucket_to_indices:
            raise ValueError("No bucket metadata found in dataset.")
        return bucket_to_indices

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        bucket_names = list(self.bucket_to_indices)
        if self.shuffle:
            rng.shuffle(bucket_names)
        for bucket_name in bucket_names:
            indices = list(self.bucket_to_indices[bucket_name])
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                yield index

    def __len__(self):
        return sum(len(indices) for indices in self.bucket_to_indices.values())
