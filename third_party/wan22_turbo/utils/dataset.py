from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from PIL import Image
import os
import torchvision.transforms.functional as TF
import pandas as pd
import cv2
import random
from pathlib import Path
import decord
from torchvision.transforms.functional import resize
from torch.utils.data.distributed import DistributedSampler


def _metadata_has_value(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value) != ""


def _metadata_int(row, key):
    value = row.get(key)
    if not _metadata_has_value(value):
        return None
    return int(float(value))


def _metadata_bucket(row, fallback_h=None, fallback_w=None):
    bucket = row.get("bucket")
    if _metadata_has_value(bucket):
        return str(bucket)
    height = _metadata_int(row, "height")
    width = _metadata_int(row, "width")
    if height is not None and width is not None:
        return "landscape" if width >= height else "portrait"
    if fallback_h is not None and fallback_w is not None:
        return "landscape" if fallback_w >= fallback_h else "portrait"
    return "landscape"


class OffsetDistributedSampler(DistributedSampler):
    def __init__(self, dataset, initial_step=0, gpu_num=4, **kwargs):
        super().__init__(dataset, **kwargs)
        if initial_step < len(dataset) // gpu_num:
            self.initial_step = initial_step
        else:
            self.initial_step = (
                (initial_step * gpu_num - len(dataset)) % len(dataset)
            ) // (gpu_num * gpu_num)
        self.first_time = True  # 标志位，表示是否是第一次加载

    def __iter__(self):
        # 获取原始索引
        indices = list(super().__iter__())

        # 如果是第一次加载，跳过前 initial_step 个索引
        if self.first_time and self.initial_step > 0:
            indices = indices[self.initial_step :]
            self.first_time = False  # 标志位设为 False，后续不再跳过

        return iter(indices)


class BucketOffsetDistributedSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        initial_step=0,
        gpu_num=1,
        rank=0,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        seed=0,
    ):
        self.dataset = dataset
        self.initial_step = max(int(initial_step), 0)
        self.gpu_num = max(int(gpu_num), 1)
        self.rank = int(rank)
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.first_time = True
        self.bucket_to_indices = self._build_bucket_indices()

    def _build_bucket_indices(self):
        if not hasattr(self.dataset, "data"):
            raise ValueError("BucketOffsetDistributedSampler requires dataset.data metadata.")
        buckets = {}
        for index in range(len(self.dataset)):
            row = self.dataset.data[index % len(self.dataset.data)]
            bucket = _metadata_bucket(row)
            buckets.setdefault(bucket, []).append(index)
        if not buckets:
            raise ValueError("No DMD bucket metadata found.")
        return buckets

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _align_bucket_for_ranks(self, indices):
        if not indices:
            return indices
        bucket_unit = self.gpu_num * self.batch_size
        remainder = len(indices) % bucket_unit
        if remainder == 0:
            return indices
        if self.drop_last and len(indices) >= bucket_unit:
            return indices[: len(indices) - remainder]
        pad = bucket_unit - remainder
        repeats = (pad // len(indices)) + 1
        return indices + (indices * repeats)[:pad]

    def __iter__(self):
        import random

        rng = random.Random(self.seed + self.epoch)
        bucket_names = list(self.bucket_to_indices)
        if self.shuffle:
            rng.shuffle(bucket_names)

        rank_indices = []
        for bucket_name in bucket_names:
            indices = list(self.bucket_to_indices[bucket_name])
            if self.shuffle:
                rng.shuffle(indices)
            indices = self._align_bucket_for_ranks(indices)
            rank_indices.extend(indices[self.rank :: self.gpu_num])

        if self.first_time and self.initial_step > 0:
            rank_indices = rank_indices[self.initial_step :]
            self.first_time = False
        return iter(rank_indices)

    def __len__(self):
        total = 0
        for indices in self.bucket_to_indices.values():
            total += len(self._align_bucket_for_ranks(list(indices))) // self.gpu_num
        return total


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextFolderDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        for file in os.listdir(data_path):
            if file.endswith(".txt"):
                with open(os.path.join(data_path, file), "r") as f:
                    text = f.read().strip()
                    self.texts.append(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"prompts": self.texts[idx], "idx": idx}


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        img = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "img", np.uint8, local_idx,
            shape=(480, 832, 3)
        )
        img = Image.fromarray(img)
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
            "img": img
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


class ODERegressionCSVDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_pair: int = int(1e8),
        num_frames=81,
        h=480,
        w=832,
        enable_orientation_buckets=False,
    ):
        self.max_pair = max_pair
        dataframe = pd.read_csv(data_path)
        dataframe["text"] = dataframe["text"].fillna("")
        self.data = dataframe.to_dict("records")
        self.log_file = "log/datasets_error_log.txt"
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.enable_orientation_buckets = enable_orientation_buckets

    def __len__(self):
        return len(self.data)

    def _target_height_width(self, sample):
        if self.enable_orientation_buckets:
            height = _metadata_int(sample, "height")
            width = _metadata_int(sample, "width")
            if height is not None and width is not None:
                return height, width
            bucket = _metadata_bucket(sample, fallback_h=self.h, fallback_w=self.w)
            if bucket == "portrait":
                return self.w, self.h
        return self.h, self.w

    def _preprocess_video(self, sample) -> torch.Tensor:
        path = sample["path"]
        num_frames = int(float(sample["num_frames"]))
        if num_frames < self.num_frames:
            raise ValueError(f"Error: num_frames < {self.num_frames}")
        frame_indices = list(range(self.num_frames))
        target_h, target_w = self._target_height_width(sample)

        if path.endswith(".mp4") or path.endswith(".mkv"):
            path = Path(path)
            video_reader = decord.VideoReader(uri=path.as_posix())
            frames = torch.tensor(
                video_reader.get_batch(frame_indices).asnumpy()
            ).float()  # [T, H, W, C]
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        else:
            image_files = sorted(os.listdir(path))
            if not os.path.isdir(path) or not image_files:
                raise ValueError("Error: Invalid images path or no images found")
            frames = []
            for frame_index in frame_indices:
                frame_path = os.path.join(path, image_files[frame_index])
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frames = np.stack(frames)
            frames = torch.from_numpy(frames).float()
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]

        video_tensor = torch.stack([resize(frame, (target_h, target_w)) for frame in frames], dim=0)
        video_tensor = video_tensor.permute(1, 0, 2, 3) / 255.0
        video_tensor = video_tensor * 2 - 1
        return video_tensor, target_h, target_w

    def __getitem__(self, index):
        sample = self.data[index]
        try:
            video, target_h, target_w = self._preprocess_video(sample)
            return {
                "prompts": sample["text"],
                "video": video,
                "height": target_h,
                "width": target_w,
                "bucket": _metadata_bucket(sample, fallback_h=target_h, fallback_w=target_w),
            }
        except Exception as e:
            # 记录错误日志
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(f"Error at index {index}: {str(e)}\n")
            print(f"Error at index {index}: {e}. Skipping this index.")
            # 跳过当前样本，返回 None 或抛出异常
            target_h, target_w = self._target_height_width(sample)
            return {
                "prompts": "",
                "video": torch.zeros((3, self.num_frames, target_h, target_w)),  # 占位符视频张量
                "height": target_h,
                "width": target_w,
                "bucket": _metadata_bucket(sample, fallback_h=target_h, fallback_w=target_w),
            }

def cycle(dl):
    while True:
        for data in dl:
            yield data

def masks_like(tensor, zero=False, generator=None, p=0.2):
    # assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[0, :] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[0, :]).exp()
                    v[0, :] = torch.zeros_like(v[0, :])
                else:
                    u[0, :] = u[0, :]
                    v[0, :] = v[0, :]
        else:
            for u, v in zip(out1, out2):
                u[0, :] = torch.zeros_like(u[0, :])
                v[0, :] = torch.zeros_like(v[0, :])

    return out1, out2
