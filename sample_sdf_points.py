
# -----------------------------------------------------------------------------
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import trimesh
import glob
import os
import numpy as np
import mesh_to_sdf
import skimage
import pyrender
import torch
import argparse
from urdf_layer import URDFLayer

parser = argparse.ArgumentParser()
parser.add_argument('--urdf_path', default='./collision_avoidance_example/xarm7_urdf/xarm7_FT_EE.urdf', type=str)
parser.add_argument('--voxel_dir', default='./panda_layer/meshes/voxel_128_xarm7', type=str)
parser.add_argument('--device', default='cuda', type=str)
args = parser.parse_args()

# dynamically scan meshes embedded directly from the declared robotURDF
robot_layer = URDFLayer(urdf_path=args.urdf_path, device=args.device, voxel_dir=args.voxel_dir)
mesh_files = robot_layer.get_mesh_paths()
mesh_names = robot_layer.get_mesh_names()

for i, (mf, mesh_name) in enumerate(zip(mesh_files, mesh_names)):
    print(mesh_name)
    mesh = trimesh.load(mf, force='mesh')
    
    scale_setting = robot_layer.meshes_info[i]['scale']
    mesh.apply_scale(scale_setting)
        
    mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)

    center = mesh.bounding_box.centroid
    scale = np.max(np.linalg.norm(mesh.vertices-center, axis=1))

    # sample points near surface (as same as deepSDF)
    near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(mesh, 
                                                      number_of_points = 500000, 
                                                      surface_point_method='scan', 
                                                      sign_method='normal', 
                                                      scan_count=100, 
                                                      scan_resolution=400, 
                                                      sample_point_count=10000000, 
                                                      normal_sample_count=100, 
                                                      min_size=0.0, 
                                                      return_gradients=False)
    # # sample points randomly within the bounding box [-1,1]
    random_points = np.random.rand(500000,3)*2.0-1.0
    random_sdf = mesh_to_sdf.mesh_to_sdf(mesh, 
                                     random_points, 
                                     surface_point_method='scan', 
                                     sign_method='normal', 
                                     bounding_radius=None, 
                                     scan_count=100, 
                                     scan_resolution=400, 
                                     sample_point_count=10000000, 
                                     normal_sample_count=100) 
    
    # save data
    data = {
        'near_points': near_points,
        'near_sdf': near_sdf,
        'random_points': random_points,
        'random_sdf': random_sdf,
        'center': center,
        'scale': scale
    }
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),f'data/sdf_points_xarm7')
    if os.path.exists(save_path) is not True:
        os.mkdir(save_path)
    np.save(os.path.join(save_path,f'voxel_128_xarm7_{mesh_name}.npy'), data)

    # # # for visualization
    # data = np.load(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),f'data/sdf_points/voxel_128_{mesh_name}.npy')), allow_pickle=True).item()
    # random_points = data['random_points']
    # random_sdf = data['random_sdf']
    # near_points = data['near_points']
    # near_sdf = data['near_sdf']
    # colors = np.zeros(random_points.shape)
    # colors[random_sdf < 0, 2] = 1
    # colors[random_sdf > 0, 0] = 1
    # cloud = pyrender.Mesh.from_points(random_points, colors=colors)
    # scene = pyrender.Scene()
    # scene.add(cloud)
    # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
