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

def render_cube(image_size=256, color=[0.7, 0.7, 1.0], device=None, output_file="images/cube_360.gif"):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Define vertices and faces for a tetrahedron.
    vertices = torch.tensor([
        [-1.0, -1.0, -1.0],  # Vertex 0
        [ 1.0, -1.0, -1.0],  # Vertex 1
        [ 1.0,  1.0, -1.0],  # Vertex 2
        [-1.0,  1.0, -1.0],  # Vertex 3
        [-1.0, -1.0,  1.0],  # Vertex 4
        [ 1.0, -1.0,  1.0],  # Vertex 5
        [ 1.0,  1.0,  1.0],  # Vertex 6
        [-1.0,  1.0,  1.0],  # Vertex 7
    ]).unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = torch.tensor([
        [0, 1, 2],  # Front face (two triangles)
        [0, 2, 3],
        [4, 5, 6],  # Back face (two triangles)
        [4, 6, 7],
        [0, 1, 5],  # Bottom face (two triangles)
        [0, 5, 4],
        [2, 3, 7],  # Top face (two triangles)
        [2, 7, 6],
        [0, 3, 7],  # Left face (two triangles)
        [0, 7, 4],
        [1, 2, 6],  # Right face (two triangles)
        [1, 6, 5],
    ]).unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

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
        distance = 5.5
        elevation = 37.5
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
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_file", type=str, default="images/cube_360.gif")
    args = parser.parse_args()
    render_cube(image_size=args.image_size, output_file=args.output_file)