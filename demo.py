import os
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from stablespanner import DemoData, Spann3R
from torch.utils.data import DataLoader

def get_args_parser():
    parser = argparse.ArgumentParser('Spann3R demo', add_help=False)
    parser.add_argument('--save_path', type=str, default='./output/demo/', help='Path to experiment folder')
    parser.add_argument('--demo_path', type=str, default='./examples/s00567', help='Path to experiment folder')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/spann3r.pth', help='ckpt path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--conf_thresh', type=float, default=1e-3, help='confidence threshold')
    parser.add_argument('--kf_every', type=int, default=10, help='map every kf_every frames')
    parser.add_argument('--vis', action='store_true', help='visualize')

    return parser

@torch.no_grad()
def main(args):

    workspace = args.save_path
    os.makedirs(workspace, exist_ok=True)

    ##### Load model
    model = Spann3R(use_feat=False).to(args.device)
    
    model.load_state_dict(torch.load(args.ckpt_path)['model'])
    model.eval()

    ##### Load dataset
    dataset = DemoData(root_dir=args.demo_path, kf_every=args.kf_every)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = dataloader.__iter__().__next__()

    ##### Inference
    for view in batch:
         view['img'] = view['img'].to(args.device, non_blocking=True)
           

    demo_name = args.demo_path.split("/")[-1]

    print(f'Started reconstruction for {demo_name}')

    start = time.time()
    preds, preds_all = model.forward(batch) 
    end = time.time()
    ordered_batch = batch
        
    fps = len(batch) / (end - start)
    

    print(f'Finished reconstruction for {demo_name}, FPS: {fps:.2f}')

    ##### Save results

    save_demo_path = osp.join(workspace, demo_name)
    os.makedirs(save_demo_path, exist_ok=True)

    pts_all = []
    images_all = []
    conf_all = []

    for j, view in enumerate(ordered_batch):
        
        image = view['img'].permute(0, 2, 3, 1).cpu().numpy()[0]
        pts = preds[j]['pts3d_in_other_view'].detach().cpu().numpy()[0]
        conf = preds[j]['conf'][0].cpu().data.numpy()


        images_all.append((image[None, ...] + 1.0)/2.0)
        pts_all.append(pts[None, ...])
        conf_all.append(conf[None, ...])
    
    images_all = np.concatenate(images_all, axis=0)
    pts_all = np.concatenate(pts_all, axis=0)
    conf_all = np.concatenate(conf_all, axis=0)

    # Save point cloud
    conf_sig_all = (conf_all-1) / conf_all

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all[conf_sig_all>args.conf_thresh].reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(images_all[conf_sig_all>args.conf_thresh].reshape(-1, 3))
    o3d.io.write_point_cloud(os.path.join(save_demo_path, f"{demo_name}_conf{args.conf_thresh}.ply"), pcd)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)