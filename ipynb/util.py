########## Import Libararies
import os
from collections import Counter
import pickle
from tqdm.notebook import tqdm, trange

import numpy as np
from skimage import measure
import scipy.ndimage

 
import open3d as o3d
import trimesh

import nibabel as nib
import csv
from compas.geometry import trimesh_remesh
from compas.datastructures import Mesh

########## Create boundary_pt and its texture into pt_{}.pickle file
def marching_cubes_3d(data):
  vertices, triangles, _, _ = measure.marching_cubes(data, level=0)
  return vertices, triangles

def voxel_to_mesh(data, flip=True):
  vertices, triangles, _, _ = measure.marching_cubes(data, level=0)
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(vertices)
  if flip:
    tri0 = triangles.copy()
    triangles[:, 0] = tri0[:, 2]
    triangles[:, 2] = tri0[:, 0]
  mesh.triangles = o3d.utility.Vector3iVector(triangles)
  mesh.compute_vertex_normals()
  
  return mesh

def mesh_to_np(mesh, shape=(256, 256, 256, 3)): 
  # Create the array
  queries = np.zeros(shape, dtype=np.float32)
  # Fill the array with values
  for i in range(shape[0]):
      for j in range(shape[1]):
          for k in range(shape[2]):
              queries[i, j, k, :] = [i, j, k]
              
  ############## mesh to numpy occupancy ##############
  mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

  # Create a scene and add the triangle mesh
  scene = o3d.t.geometry.RaycastingScene()
  _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

  occupancy = scene.compute_occupancy(queries)
  return occupancy.numpy()

def edge_adjusted_mesh(mesh, target_length = 1):

  mesh_tri = Mesh.from_vertices_and_faces(vertices=np.asarray(mesh.vertices).tolist(),
                        faces=np.asarray(mesh.triangles).tolist())                       
  vert, fac = trimesh_remesh(mesh_tri.to_vertices_and_faces(), target_edge_length=target_length)
  print(len(vert))
  mesh_avg= o3d.geometry.TriangleMesh()
  mesh_avg.vertices = o3d.utility.Vector3dVector(np.asarray(vert))
  mesh_avg.triangles = o3d.utility.Vector3iVector(np.asarray(fac))
  return mesh_avg

def sdf_query(mesh, query_points):

  mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

  # Create a scene and add the triangle mesh
  scene = o3d.t.geometry.RaycastingScene()
  _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
  
  sdf = scene.compute_signed_distance(query_points)
  # print(query_points.shape, occupancy.shape)
  # We can visualize a slice of the distance field directly with matplotlib
  # plt.imshow(occupancy.numpy()[:, :, 70])
  #print(np.shape(sdf.numpy()))
  return sdf.numpy()

def find_nearest_points_indices(point_cloud1, point_cloud2):
    # Calculate the distance matrix
    dist_matrix = np.linalg.norm(point_cloud1[:, None, :] - point_cloud2[None, :, :], axis=-1)
    
    # Find the index of the nearest point for each point in point_cloud1
    nearest_indices = np.argmin(dist_matrix, axis=1)
    
    return nearest_indices

def read_sub_from_csv(file_path, flag = "LBC"):
    data = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index==0: continue
            else:
                if flag in row[0]:
                    id= row[0]
                    data.append(id)
    return data

def read_dict_from_csv(file_path, flag = "LBC"):
    data = {}
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index==0: continue
            else:
                if flag in row[0]:
                    id= row[0]
                    data[id] = row
    return data

def pt_to_tex(boundary_pt, aseg_path, tex_label = [10, 11, 43]):
  
    aseg_peri = np.array(nib.load(aseg_path).dataobj)
    aseg_peri[(aseg_peri!=tex_label[0])&(aseg_peri!=tex_label[1])&(aseg_peri!=tex_label[2])]=0
    aseg_peri[aseg_peri==tex_label[0]]=1
    aseg_peri[aseg_peri==tex_label[1]]=2
    aseg_peri[aseg_peri==tex_label[2]]=3
    texture = np.zeros(boundary_pt.shape)
    print(texture.shape)
    
    for n in range(boundary_pt.shape[0]):
      a= []
      x, y, z = (boundary_pt[n][0]), (boundary_pt[n][1]), (boundary_pt[n][2])
      peri_list = [2.5, 1.5, 0.5, -0.5, -1.5, -2.5]
      for i in peri_list:
        for j in peri_list:
          for k in peri_list:
            tex = aseg_peri[int(x+i),int(y+j),int(z+k)]
            if tex!=0:
              a.append(tex)
      if a:
        counts = Counter(a)
        most_common_value = max(counts, key=lambda x: counts[x] if (x!=0) else -1)
        texture[n]=int(most_common_value)

    return  texture

def index_of_tex(texture_new, tri, tex_label):
    tex_index = np.argwhere(texture_new[:,0]==tex_label).flatten()
    mask = np.isin(tri, tex_index).any(axis=1)
    indices = np.where(mask)[0]
    return indices
  
def regist_mesh_pcd(src_pcd, tar_pcd, src_mesh=None, return_reg_p2p=False, normal=True):
  if normal ==False:
      reg_p2p = o3d.pipelines.registration.registration_icp(  # source, target
		src_pcd, tar_pcd, max_correspondence_distance=10,  # Adjust the distance threshold as needed
		estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
  else:
    reg_p2p = o3d.pipelines.registration.registration_icp(  # source, target
      src_pcd, tar_pcd, max_correspondence_distance=10,  # Adjust the distance threshold as needed
      estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
      criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
    
  src_pcd.transform(reg_p2p.transformation)
  if src_mesh:src_mesh.transform(reg_p2p.transformation)
  if return_reg_p2p:
    return reg_p2p.transformation
  
def voxel_to_bd_pcd(voxel, threshold = 8):
  kernel = np.ones((2,2,2))
  voxel_bd = scipy.ndimage.convolve((voxel).astype(int), kernel, mode='constant', cval=0.0)
  voxel_v = np.argwhere((voxel_bd<threshold)&(voxel_bd>0)).astype(float) + 0.5
  voxel_pcd = o3d.geometry.PointCloud()
  voxel_pcd.points = o3d.utility.Vector3dVector(voxel_v)
  return voxel_pcd

def apply_morphology(grid, structure_size=3):
    # Define a structuring element (cube)
    struct_elem = np.ones((structure_size, structure_size, structure_size))
    # Apply dilation followed by erosion (Closing operation)
    dilated_grid = ndimage.binary_dilation(grid, structure=struct_elem)
    eroded_grid = ndimage.binary_erosion(dilated_grid, structure=struct_elem)
    return eroded_grid
