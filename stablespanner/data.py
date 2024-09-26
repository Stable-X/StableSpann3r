import os
import numpy as np
import torch
from PIL import Image, ImageOps
from typing import List, Dict, Union, Tuple
from torchvision import transforms

class DemoData:
    def __init__(self, root_dir: str, resolution: Tuple[int, int] = (224, 224), kf_every: int = 10):
        self.root_dir = root_dir
        self.kf_every = kf_every
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.image_files = self._get_image_files()

    def _get_image_files(self) -> List[str]:
        supported_extensions = (".jpg", ".jpeg", ".png", ".heic", ".heif")
        all_files = [f for f in sorted(os.listdir(self.root_dir)) 
                     if f.lower().endswith(supported_extensions)]
        # Apply keyframe sampling
        return all_files[::self.kf_every]

    def __len__(self) -> int:
        return 1  # Only one sequence

    def _process_image(self, img_path: str) -> Dict[str, Union[torch.Tensor, List[int], str]]:
        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        W1, H1 = img.size

        # Convert to tensor
        img_tensor = self.transform(img)
        
        # Normalize
        img_tensor = self.normalize(img_tensor)

        W2, H2 = self.resolution
        print(f" - Processing {img_path} with resolution {W1}x{H1} --> {W2}x{H2}")

        return {
            "img": img_tensor,  # Add batch dimension
            "true_shape": torch.tensor([H2, W2], dtype=torch.int32),
            "idx": 0,  # This will be updated in __getitem__
            "instance": os.path.basename(img_path),
            "label": img_path,
        }

    def __getitem__(self, idx: int) -> List[Dict[str, Union[torch.Tensor, List[int], str]]]:
        views = []

        for i, img_file in enumerate(self.image_files):
            img_path = os.path.join(self.root_dir, img_file)
            view = self._process_image(img_path)
            view["idx"] = i
            views.append(view)

        assert views, f"No images found in {self.root_dir}"
        print(f" (Found {len(views)} images)")
        return views