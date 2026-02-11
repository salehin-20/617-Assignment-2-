import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


class GarbageDataset(Dataset):
    """
    Multimodal dataset:
      - Image: loaded from folders Black/Blue/Green/Other
      - Text: derived from filename (underscores -> spaces) and encoded as char IDs
      - Label: 0..3
    Returns: (image, text_tensor, label_tensor)
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        max_text_len: int = 32,
        include_heic: bool = True,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.max_text_len = max_text_len

        self.classes = ["Black", "Blue", "Green", "Other"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        exts = list(IMG_EXTENSIONS)
        if not include_heic:
            exts = [e for e in exts if e != ".heic"]
        self.img_exts = tuple(exts)

        self.samples: List[Tuple[str, str, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            for fname in os.listdir(cls_path):
                # Skip junk files (e.g., Zone.Identifier) and keep only images
                if not fname.lower().endswith(self.img_exts):
                    continue

                img_path = os.path.join(cls_path, fname)
                text = os.path.splitext(fname)[0].replace("_", " ")
                label = self.class_to_idx[cls]
                self.samples.append((img_path, text, label))

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_text_char_ids(self, text: str) -> torch.Tensor:
        """
        Simple baseline: character-level encoding.
        Each character -> ord(c). Pad with 0 to max_text_len.
        """
        text = text.lower().strip()[: self.max_text_len]
        ids = [ord(c) for c in text]
        if len(ids) < self.max_text_len:
            ids += [0] * (self.max_text_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        img_path, text, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text_tensor = self._encode_text_char_ids(text)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, text_tensor, label_tensor
