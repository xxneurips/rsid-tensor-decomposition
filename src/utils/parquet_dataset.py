"""Iterable map-style dataset for HF imagenet-1k parquet shards.

Streams row groups sequentially per shard, caches the current row group, and
randomly samples within it. This is much faster than per-sample random access
which forces re-reading entire row groups.
"""
import os
import io
import random
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


class ParquetImageNetDataset(IterableDataset):
    """Iterable dataset for HF imagenet-1k parquet shards (split = 'train' or 'validation')."""

    def __init__(self, parquet_dir, split, transform, shuffle=True):
        self.parquet_dir = parquet_dir
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        prefix = "train-" if split == "train" else "validation-"
        all_files = []
        for root, dirs, files in os.walk(parquet_dir):
            for fn in files:
                if fn.startswith(prefix) and fn.endswith(".parquet"):
                    all_files.append(os.path.join(root, fn))
        all_files.sort()
        if not all_files:
            raise FileNotFoundError(f"No {prefix}*.parquet under {parquet_dir}")
        self.shards = all_files
        # Total length cached
        self._length = sum(pq.ParquetFile(f).metadata.num_rows for f in all_files)
        print(f"[ParquetImageNet/{split}] {len(self.shards)} shards, {self._length} samples")

    def __len__(self):
        return self._length

    def __iter__(self):
        winfo = get_worker_info()
        shards = list(self.shards)
        if winfo is not None:
            shards = shards[winfo.id::winfo.num_workers]
        if self.shuffle:
            random.shuffle(shards)
        for shard_path in shards:
            pf = pq.ParquetFile(shard_path)
            rg_indices = list(range(pf.num_row_groups))
            if self.shuffle:
                random.shuffle(rg_indices)
            for rg_idx in rg_indices:
                rg = pf.read_row_group(rg_idx, columns=["image", "label"])
                imgs = rg.column("image").to_pylist()
                labels = rg.column("label").to_pylist()
                indices = list(range(len(imgs)))
                if self.shuffle:
                    random.shuffle(indices)
                for i in indices:
                    img_struct = imgs[i]
                    if isinstance(img_struct, dict):
                        img_bytes = img_struct.get("bytes")
                        if img_bytes is None and img_struct.get("path"):
                            with open(img_struct["path"], "rb") as fp:
                                img_bytes = fp.read()
                    else:
                        img_bytes = img_struct
                    try:
                        with Image.open(io.BytesIO(img_bytes)) as im:
                            im = im.convert("RGB")
                            if self.transform is not None:
                                im = self.transform(im)
                        yield im, int(labels[i])
                    except Exception:
                        continue
