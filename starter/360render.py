"""
Usage:
    python -m starter.360render --num_frames 36
"""
import argparse

import imageio
import numpy as np
import pytorch3d
import torch
from PIL import Image
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_360(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/360render_cow.gif",
):
    if device is None:
        device = get_device()
    
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)
    cow_path = "data/cow.obj"
    color = [0.7, 0.7, 1]
    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    camera_positions = []
    for frame_idx in range(num_frames):
        azimuth = 360 * frame_idx / num_frames
        distance = 3.0
        elevation = 30.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=device)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output_file", type=str, default="images/360render_cow.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    render_360(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )