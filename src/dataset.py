import os
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic")


class GarbageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path like data/dataset/assignment1
        """
        self.root_dir = root_dir
        self.transform = transform

        self.classes = ["Black", "Blue", "Green", "Other"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []

        self._load_samples()

    def _load_samples(self):
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            for fname in os.listdir(cls_path):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    img_path = os.path.join(cls_path, fname)

                    # text from filename
                    text = os.path.splitext(fname)[0].replace("_", " ")

                    label = self.class_to_idx[cls]
                    self.samples.append((img_path, text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text, label = self.samples[idx]
