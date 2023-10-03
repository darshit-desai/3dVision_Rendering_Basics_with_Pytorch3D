"""
Usage:
    python -m starter.cow_retexture --num_frames 36
"""
import argparse

import imageio
import numpy as np
import pytorch3d
import torch
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def retexture_360(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/retexture_360render.gif",
):
    if device is None:
        device = get_device()
    cow_path="data/cow.obj"
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    color1 = [0.0, 0.0, 0.5]
    color2 = [1.0, 1.0, 0.0]
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    z_min = vertices[:,:,2].min()
    z_max = vertices[:,:,2].max()
    alpha = (vertices[:, :, 2] - z_min) / (z_max - z_min)
    new_colors = alpha[:, :, None] * torch.tensor(color2) + (1 - alpha[:, :, None]) * torch.tensor(color1)
    textures = pytorch3d.renderer.TexturesVertex(new_colors)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=textures,
    )
    mesh = mesh.to(device)
    # Prepare the camera:
    distance = 3.0
    elevation = 30.0
    azimuth = 250.0
    R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R, T=T, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # Render the retextured cow.
    image_retexture = renderer(mesh, cameras=cameras, lights=lights)
    image_retexture = image_retexture.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    camera_positions = []
    for frame_idx in range(num_frames):
        azimuth = 360 * frame_idx / num_frames
        distance = 3.0
        elevation = 30.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=device,degrees=True)
        camera_positions.append((R,T))

    renders = []
    for R,T in tqdm(camera_positions):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration, loop=0)
    return image_retexture

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output_file", type=str, default="images/retexture_360render.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    image = retexture_360(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )
    output_path = "images/cow_retexture.jpg"
    plt.imsave(output_path, image)