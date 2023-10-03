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


image_size = 256
background_color=(1, 1, 1)
num_frames = 10
output_file1="images/PC1_360render.gif"
output_file2="images/PC2_360render.gif"
duration = 3.0
rotation_angle_radians = np.pi
rgbd_plant = load_rgbd_data(path="data/rgbd_data.pkl")
points1 = []
color_rgb1 = []
points1, color_rgb1  = unproject_depth_image(image=torch.tensor(rgbd_plant['rgb1']), mask=torch.tensor(rgbd_plant['mask1']),
                                             depth=torch.tensor(rgbd_plant['depth1']), camera=rgbd_plant['cameras1'])

device = get_device()
renderer = get_points_renderer(image_size=image_size, background_color=background_color)
rotation_matrix = torch.tensor([[np.cos(rotation_angle_radians), -np.sin(rotation_angle_radians), 0.0],
                                [np.sin(rotation_angle_radians), np.cos(rotation_angle_radians), 0.0],
                                [0.0, 0.0, 1.0]])
rot_points1=torch.matmul(torch.tensor(points1), rotation_matrix.float())
point_cloud1 = pytorch3d.structures.Pointclouds(points=rot_points1.to(device).unsqueeze(0), features=torch.tensor(color_rgb1).to(device).unsqueeze(0))
camera_positions = []
for frame_idx in range(num_frames):
    azimuth = 360 * frame_idx / num_frames
    distance = 6.0
    elevation = 0.0
    R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=device, degrees=True)
    camera_positions.append((R,T))

renders1 = []
for R,T in tqdm(camera_positions):
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud1, cameras=cameras)
    rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
    renders1.append(rend)
images1 = []
for i, r in enumerate(renders1):
    image = Image.fromarray((r * 255).astype(np.uint8))
    images1.append(np.array(image))
imageio.mimsave(output_file1, images1, duration=duration, loop=0)

points2 = []
color_rgb2 = []
points2, color_rgb2 = unproject_depth_image(image=torch.tensor(rgbd_plant['rgb2']), mask=torch.tensor(rgbd_plant['mask2']), 
                                            depth=torch.tensor(rgbd_plant['depth2']), camera=rgbd_plant['cameras2'])

rot_points2=torch.matmul(torch.tensor(points2), rotation_matrix.float())
point_cloud2 = pytorch3d.structures.Pointclouds(points=rot_points2.to(device).unsqueeze(0), features=torch.tensor(color_rgb2).to(device).unsqueeze(0))

renders2 = []
for R,T in tqdm(camera_positions):
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud2, cameras=cameras)
    rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
    renders2.append(rend)
images2 = []
for i, r in enumerate(renders2):
    image = Image.fromarray((r * 255).astype(np.uint8))
    images2.append(np.array(image))
imageio.mimsave(output_file2, images2, duration=duration, loop=0)

rotated_all_points = torch.cat([rot_points1, rot_points2], dim=0)
all_colors = torch.cat([torch.tensor(color_rgb1), torch.tensor(color_rgb2)], dim=0)
point_cloud_union = pytorch3d.structures.Pointclouds(points=rotated_all_points.to(device).unsqueeze(0), features=all_colors.to(device).unsqueeze(0))

render_full = []
for R,T in tqdm(camera_positions):
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud_union, cameras=cameras)
    rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
    render_full.append(rend)
images_full = []
for i, r in enumerate(render_full):
    image = Image.fromarray((r * 255).astype(np.uint8))
    images_full.append(np.array(image))
imageio.mimsave("images/PCUnion_360render.gif", images_full, duration=duration, loop=0)







