import gradio as gr
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
import uuid
import time

from spann3r.model import Spann3R
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs

# Initialize the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = Spann3R(use_feat=False).to(DEVICE)
model.load_state_dict(torch.load("path/to/spann3r.pth", map_location=DEVICE)['model'])
model.eval()

def process_images(image_paths):
    imgs_all = []
    for idx, img_path in enumerate(image_paths):
        img = torch.from_numpy(np.array(img_path)).permute(2, 0, 1).float() / 255.0
        imgs_all.append(
            dict(
                img=img.unsqueeze(0),
                true_shape=torch.tensor(img.shape[1:]).unsqueeze(0),
                idx=idx,
                instance=str(idx)
            )
        )
    return imgs_all

@torch.no_grad()
def reconstruct_3d(image_paths):
    imgs_all = process_images(image_paths)
    
    start = time.time()
    
    pairs = make_pairs(imgs_all, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model.dust3r, DEVICE, batch_size=2, verbose=True)
    preds, preds_all, idx_used = model.offline_reconstruction(imgs_all, output) 
    
    end = time.time()
    fps = len(imgs_all) / (end - start)
    
    pts_all = []
    images_all = []
    conf_all = []
    
    for j, pred in enumerate(preds):
        image = imgs_all[idx_used[j]]['img'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        pts = pred['pts3d' if j==0 else 'pts3d_in_other_view'].cpu().numpy()[0]
        conf = pred['conf'][0].cpu().data.numpy()
        
        images_all.append(image)
        pts_all.append(pts)
        conf_all.append(conf)
    
    pts_all = np.concatenate(pts_all, axis=0)
    images_all = np.stack(images_all, axis=0)
    conf_all = np.concatenate(conf_all, axis=0)
    
    conf_sig_all = (conf_all-1) / conf_all
    conf_thresh = 1e-3
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all[conf_sig_all>conf_thresh].reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(images_all[conf_sig_all>conf_thresh].reshape(-1, 3))
    
    output_dir = Path("/tmp/spann3r_output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{uuid.uuid4()}_conf{conf_thresh}.ply"
    o3d.io.write_point_cloud(str(output_file), pcd)
    
    return str(output_file), fps

def spann3r_demo(images):
    if not images:
        return "Please upload at least two images."
    
    image_paths = [image.name for image in images]
    try:
        output_file, fps = reconstruct_3d(image_paths)
        return f"3D reconstruction completed. Output saved to: {output_file}. FPS: {fps:.2f}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=spann3r_demo,
    inputs=gr.File(file_count="multiple"),
    outputs="text",
    title="Spann3R: 3D Reconstruction Demo",
    description="Upload multiple images to reconstruct a 3D scene using Spann3R.",
)

# Launch the demo
iface.launch()