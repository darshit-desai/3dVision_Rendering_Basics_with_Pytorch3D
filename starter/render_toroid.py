import argparse
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image

from starter.render_generic import load_rgbd_data


def render_toroid(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()
    toroid_radius = 1.0
    tube_radius = 0.5
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (torch.tensor(toroid_radius)+torch.tensor(tube_radius)*torch.cos(Theta)) * torch.cos(Phi)
    y = (torch.tensor(toroid_radius)+torch.tensor(tube_radius)*torch.cos(Theta)) * torch.sin(Phi)
    z = torch.sin(Theta) * torch.tensor(tube_radius)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())
    toroid_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend_pic = renderer(toroid_point_cloud, cameras=cameras)
    rend_pic = rend_pic[0, ..., :3].cpu().numpy()
    num_frames = 10
    render_full = []
    camera_positions = []
    for frame_idx in range(num_frames):
        azimuth = 360 * frame_idx / num_frames
        distance = 6.0
        elevation = 0.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=device, degrees=True)
        camera_positions.append((R,T))
    for R,T in tqdm(camera_positions):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(toroid_point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        render_full.append(rend)
    images = []
    for i, r in enumerate(render_full):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave("images/toroid_parametric_360render.gif", images, duration=3.0, loop=0)
    return rend_pic

def render_toroid_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -2.5
    max_value = 2.5
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    toroid_radius = torch.tensor(1.0)
    tube_radius = torch.tensor(0.5)
    # voxels = (X ** 2 + Y ** 2 + Z**2 + torch.tensor(toroid_radius)**2 - torch.tensor(tube_radius)**2)**2 - 4*torch.tensor(toroid_radius)**2*(X**2 + Y**2)
    voxels = (torch.sqrt(X**2 + Y**2) - toroid_radius)**2 + Z**2 - tube_radius**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    alpha = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    colors = alpha[:, 2][:, None] * torch.tensor([1.0, 0.0, 0.0]) + (1 - alpha[:, 2][:, None]) * torch.tensor([0.0, 0.0, 1.0])
    textures = pytorch3d.renderer.TexturesVertex(colors[None, ...])

    toroid_mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend_pic = renderer(toroid_mesh, cameras=cameras, lights=lights)
    rend_pic = rend_pic[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    num_frames = 10
    render_full = []
    camera_positions = []
    for frame_idx in range(num_frames):
        azimuth = 360 * frame_idx / num_frames
        distance = 6.0
        elevation = 30.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=device, degrees=True)
        camera_positions.append((R,T))
    for R,T in tqdm(camera_positions):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(toroid_mesh, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        render_full.append(rend)
    images = []
    for i, r in enumerate(render_full):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave("images/toroid_implicit_360render.gif", images, duration=3.0, loop=0)
    return rend_pic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["parametric", "implicit"],
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "parametric":
        image = render_toroid(image_size=args.image_size, num_samples=args.num_samples)
        output_file = "images/toroid_"+args.render+".jpg"
        plt.imsave(output_file, image)
    elif args.render == "implicit":
        image = render_toroid_mesh(image_size=args.image_size)
        output_file = "images/toroid_"+args.render+".jpg"
        plt.imsave(output_file, image)
    else:
        raise Exception("Did not understand {}".format(args.render))
