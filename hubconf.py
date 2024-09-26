import torch
import os
from typing import Optional
import numpy as np
import torch.nn.functional as F

dependencies = ["torch", "numpy"]

def _load_state_dict(local_file_path: Optional[str] = None):
    if local_file_path is not None and os.path.exists(local_file_path):
        # Load state_dict from local file
        state_dict = torch.load(local_file_path, map_location=torch.device("cpu"))
    else:
        # Load state_dict from the default URL
        file_name = "spann3r.pth"
        url = f"https://huggingface.co/camenduru/Spann3R/resolve/main/spann3r.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, file_name=file_name, map_location=torch.device("cpu"))

    return state_dict['model']

class Predictor:
    def __init__(self, model) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
    
    def infer(self, batch):
        for view in batch:
            view['img'] = view['img'].to(self.device, non_blocking=True)
        
        with torch.no_grad():
            preds, preds_all = self.model.forward(batch)
        
        return preds, preds_all

def Spann3R(local_file_path: Optional[str] = None):
    from stablespanner import Spann3R as Spann3RModel

    state_dict = _load_state_dict(local_file_path)
    model = Spann3RModel(use_feat=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return Predictor(model)

def _test_run():
    import argparse
    from torch.utils.data import DataLoader
    from stablespanner import DemoData

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--demo_path", type=str, default='./examples/s00567', help="demo data path")
    parser.add_argument("--kf_every", type=int, default=10, help="keyframe interval")
    args = parser.parse_args()

    predictor = torch.hub.load(".", "Spann3R", source="local", trust_repo=True)
    
    dataset = DemoData(root_dir=args.demo_path, kf_every=args.kf_every)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    preds, preds_all = predictor.infer(batch)
    
    print(f"Predictions shape: {len(preds)}")
    print(f"All predictions shape: {len(preds_all)}")

if __name__ == "__main__":
    _test_run()