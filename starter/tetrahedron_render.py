"""
Sample code to render a cow.

Usage:
    python -m starter.tetrahedron_render --image_size 256 --output_file images/tetrahedron_360.gif
"""
import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import os
import numpy as np
from PIL import Image
import imageio
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer

def render_tetrahedron(image_size=256, color=[1.0, 0.0, 0.0], device=None, output_file="images/tetrahedron_360.gif"):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Define vertices and faces for a tetrahedron.
    vertices = torch.tensor([
        [0, 0, 0],  # Vertex 0
        [1.0, 0, 0],  # Vertex 1
        [0.5, 0, 0.8660254037844386],  # Vertex 2
        [0.5, 0.8660254037844386, 0.8660254037844386],  # Vertex 3
    ]).unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)

    faces = torch.tensor([
        [0, 1, 2],  # Triangle 0 (vertices 0, 1, 2)
        [0, 1, 3],  # Triangle 1 (vertices 0, 1, 3)
        [1, 2, 3],  # Triangle 2 (vertices 1, 2, 3)
        [2, 0, 3],  # Triangle 3 (vertices 2, 0, 3)
    ], dtype=torch.long).unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    
    # Create a single-color texture for the tetrahedron.
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    # Prepare the camera:
    num_frames = 36  # Number of frames for a full rotation
    camera_positions = []
    duration = 3
    for frame_idx in range(num_frames):
        azimuth = 360.0 * frame_idx / num_frames
        distance = 3.0
        elevation = 30
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=device, degrees=True)
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
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_file", type=str, default="images/360render_tetrahedron.gif")
    args = parser.parse_args()
    render_tetrahedron(image_size=args.image_size, output_file=args.output_file)


