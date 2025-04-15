from tqdm import tqdm
import os
import math
import pickle
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import datasets


class SingleClassDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]


class ConceptDistillationTrainer:
    def __init__(
        self, pkl_dir, mapping_file, base_dataset, batch_size=32, num_workers=0
    ):
        self.num_classes = 1000
        self.batch_size = batch_size

        self.teacher_concepts = {}
        for class_id in range(self.num_classes):
            pkl_path = os.path.join(pkl_dir, f"{class_id}.pkl")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            self.teacher_concepts[class_id] = data["W"]

        mapping_arr = np.load(mapping_file, allow_pickle=True).item()

        self.datasets = {}
        self.dataloaders = {}
        for class_id in range(self.num_classes):
            indices_for_class = mapping_arr[class_id]
            self.datasets[class_id] = SingleClassDataset(
                base_dataset, indices_for_class
            )

            loader = DataLoader(
                self.datasets[class_id],
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
            )
            self.dataloaders[class_id] = iter(loader)

        self.OrigRemLen = {
            class_id: len(mapping_arr[class_id]) for class_id in range(self.num_classes)
        }
        self.dataset_size = sum(self.OrigRemLen.values())

    def train(self, epochs=1):
        for epoch in range(epochs):
            print(f"=== EPOCH {epoch+1}/{epochs} ===")

            RemLen = dict(self.OrigRemLen)
            valid_class_ids = list(RemLen.keys())

            num_batches = sum(
                math.ceil(len(self.datasets[class_id]) / self.batch_size)
                for class_id in valid_class_ids
            )

            for batch_idx in tqdm(range(num_batches)):
                chosen_class = random.choice(valid_class_ids)

                try:
                    batch = next(self.dataloaders[chosen_class])
                except StopIteration:
                    raise RuntimeError(
                        f"DataLoader for class {chosen_class} exhausted! "
                        "Please check your dataset and mapping logic once again."
                    )

                W = self.teacher_concepts[chosen_class]

                # Training step (abstracted)
                # Here you would typically call your model's training method
                # For example:
                # loss = model.train_step(batch, W)
                # loss.backward()
                # optimizer.step()

                RemLen[chosen_class] = max(0, RemLen[chosen_class] - self.batch_size)
                if RemLen[chosen_class] == 0:
                    valid_class_ids.remove(chosen_class)

        assert len(valid_class_ids) == 0, "Some classes still have remaining samples!"


if __name__ == "__main__":
    # Example usage
    model = (
        timm.create_model(
            "resnet50.a2_in1k",
            pretrained=True,
            cache_dir=f"/scratch/swayam/timm_cache/",
        )
        .eval()
        .requires_grad_(False)
        .to("cuda")
    )

    config = resolve_data_config({}, model=model)
    imagenet_transform = create_transform(**config)

    base_dataset = datasets.ImageNet(
        root="/scratch/swayam/imagenet_data/imagenet",
        split="val",
        transform=imagenet_transform,
    )

    pkl_dir = "/scratch/swayam/rsvc-exps/outputs/data/dn=in_spl=val_ni=100_seed=0/r50_ckpt=None/ps=64_flv=v1_igs=c/dm=nmf_nc=10_seed=0/concepts/layer4.2.act3"
    mapping_file = "restructured_map_val.npy"

    trainer = ConceptDistillationTrainer(pkl_dir, mapping_file, base_dataset)
    trainer.train()
