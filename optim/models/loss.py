import torch.nn as nn
import torch 
import torch.nn.functional as F 

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance,  mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency, point_mesh_face_distance)

import numpy as np
from itertools import product, combinations, chain
from scipy.spatial import ConvexHull

import time 
from collections import Counter
import torch.nn.functional as F
from models.loss_3d import mesh_edge_var_loss
from models.point_mesh_loss import point_mesh_face_distance2
# from models.point_mesh_loss3 import point_mesh_face_distance3
# from models.normal_L2_loss import normal_L2_loss

def lv_criterion(verts, lv_target, lv_tri, lda=[3,0.5,1500,5,1,1,1]):
    loss=0.0

    # cf loss
    pred_lv_mesh = Meshes(verts=list(verts), faces=list(lv_tri))
    pred_lv_points = sample_points_from_meshes(pred_lv_mesh, lv_target.shape[1])
    lv_chamfer_loss =  chamfer_distance(pred_lv_points, lv_target)[0]
    
    # point-mesh loss
    lv_pointclouds = Pointclouds(points=lv_target)
    lv_point_mesh_dist_loss = point_mesh_face_distance2(pred_lv_mesh, lv_pointclouds)

    # Regularization loss
    lv_edge_loss = mesh_edge_var_loss(pred_lv_mesh)
    lv_laplacian_loss =  mesh_laplacian_smoothing(pred_lv_mesh, method="cotcurv")
    lv_normal_consistency_loss = mesh_normal_consistency(pred_lv_mesh)
    
    
    loss =  lda[0]*  (lv_point_mesh_dist_loss)\
    + lda[1] * (lv_chamfer_loss)\
    + lda[2] * (lv_edge_loss)\
    + lda[3] * (lv_laplacian_loss)\
    + lda[4] * (lv_normal_consistency_loss)\

    log = {"loss": loss,
        "point_mesh_dist_loss": lv_point_mesh_dist_loss.detach(),
        "chamfer_loss": lv_chamfer_loss.detach(),
        "edge_loss": lv_edge_loss.detach(),
        "laplacian_loss": lv_laplacian_loss.detach(),
        "normal_consistency_loss": lv_normal_consistency_loss.detach()
        }
    
    return loss, log

def hippo_criterion(verts, target, faces, lda=[3,0.5,1500,5,1,1,1]):
    loss=0.0

    # cf loss
    pred_mesh = Meshes(verts=list(verts), faces=list(faces))
    pred_points = sample_points_from_meshes(pred_mesh, target.shape[1])
    chamfer_loss =  chamfer_distance(pred_points, target)[0]
    
    # point-mesh loss
    pointclouds = Pointclouds(points=target)
    point_mesh_dist_loss = point_mesh_face_distance(pred_mesh, pointclouds)

    # Regularization loss
    edge_loss = mesh_edge_var_loss(pred_mesh)
    laplacian_loss =  mesh_laplacian_smoothing(pred_mesh, method="uniform")
    normal_consistency_loss = mesh_normal_consistency(pred_mesh)
    
    
    loss =  lda[0]*  (point_mesh_dist_loss)\
    + lda[1] * (chamfer_loss)\
    + lda[2] * (edge_loss)\
    + lda[3] * (laplacian_loss)\
    + lda[4]* (normal_consistency_loss)\

    log = {"loss": loss,
        "point_mesh_dist_loss": point_mesh_dist_loss.detach(),
        "chamfer_loss": chamfer_loss.detach(),
        "edge_loss": edge_loss.detach(),
        "laplacian_loss": laplacian_loss.detach(),
        "normal_consistency_loss": normal_consistency_loss.detach()
        }
    
    return loss, log

def iccvw_criterion(verts, hippo_target, hippo_faces, thal_target, thal_faces, lda=[3,0.5,1500,5,1,1,1]):
    loss=0.0

    # cf loss
    hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
    hippo_pred_points = sample_points_from_meshes(hippo_pred_mesh, hippo_target.shape[1])
    hippo_chamfer_loss =  chamfer_distance(hippo_pred_points, hippo_target)[0]
    
    thal_pred_mesh = Meshes(verts=list(verts), faces=list(thal_faces))
    thal_pred_points = sample_points_from_meshes(thal_pred_mesh, thal_target.shape[1])
    thal_chamfer_loss =  chamfer_distance(thal_pred_points, thal_target)[0]

    # point-mesh loss
    hippo_pointclouds = Pointclouds(points=hippo_target)
    hippo_point_mesh_dist_loss = point_mesh_face_distance(hippo_pred_mesh, hippo_pointclouds)
    thal_pointclouds = Pointclouds(points=thal_target)
    thal_point_mesh_dist_loss = point_mesh_face_distance(thal_pred_mesh, thal_pointclouds)

    # Regularization loss
    hippo_edge_loss = mesh_edge_var_loss(hippo_pred_mesh)
    hippo_laplacian_loss =  mesh_laplacian_smoothing(hippo_pred_mesh, method="uniform")
    hippo_normal_consistency_loss = mesh_normal_consistency(hippo_pred_mesh)
    
    thal_edge_loss = mesh_edge_var_loss(thal_pred_mesh)
    thal_laplacian_loss =  mesh_laplacian_smoothing(thal_pred_mesh, method="uniform")
    thal_normal_consistency_loss = mesh_normal_consistency(thal_pred_mesh)
    

    loss =  lda[0]*  (hippo_point_mesh_dist_loss + thal_point_mesh_dist_loss)*0.5\
    + lda[1] * (hippo_chamfer_loss + thal_chamfer_loss)*0.5\
    + lda[2] * (hippo_edge_loss + thal_edge_loss)*0.5\
    + lda[3] * (hippo_laplacian_loss + thal_laplacian_loss)*0.5\
    + lda[4]* (hippo_normal_consistency_loss + thal_normal_consistency_loss)*0.5\

    log = {"loss": loss,
        "point_mesh_dist_loss": hippo_point_mesh_dist_loss.detach() + thal_point_mesh_dist_loss.detach(),
        "chamfer_loss": hippo_chamfer_loss.detach() + thal_chamfer_loss.detach(),
        "edge_loss": hippo_edge_loss.detach() + thal_edge_loss.detach(),
        "laplacian_loss": hippo_laplacian_loss.detach() + thal_laplacian_loss.detach(),
        "normal_consistency_loss": hippo_normal_consistency_loss.detach() + thal_normal_consistency_loss.detach()
        }
    
    return loss, log


def lv_hippo_criterion(verts, lv_target, hippo_target, lv_tri, hippo_tri, lda=[3,0.5,1500,5,1,1,1]):
    loss=0.0

    # cf loss
    pred_lv_mesh = Meshes(verts=list(verts), faces=list(lv_tri))
    pred_lv_points = sample_points_from_meshes(pred_lv_mesh, lv_target.shape[1])
    lv_chamfer_loss =  chamfer_distance(pred_lv_points, lv_target)[0]
    pred_hippo_mesh = Meshes(verts=list(verts), faces=list(hippo_tri)) 
    pred_hippo_points = sample_points_from_meshes(pred_hippo_mesh, hippo_target.shape[1])
    hippo_chamfer_loss =  chamfer_distance(pred_hippo_points, hippo_target)[0]
    
    # point-mesh loss
    lv_pointclouds = Pointclouds(points=lv_target)
    lv_point_mesh_dist_loss = point_mesh_face_distance2(pred_lv_mesh, lv_pointclouds)
    hippo_pointclouds = Pointclouds(points=hippo_target)
    hippo_point_mesh_dist_loss = point_mesh_face_distance2(pred_hippo_mesh, hippo_pointclouds)

    # Regularization loss
    lv_edge_loss = mesh_edge_var_loss(pred_lv_mesh)
    hippo_edge_loss = mesh_edge_var_loss(pred_hippo_mesh)
    lv_laplacian_loss =  mesh_laplacian_smoothing(pred_lv_mesh, method="cotcurv")
    hippo_laplacian_loss =  mesh_laplacian_smoothing(pred_hippo_mesh, method="cotcurv")
    lv_normal_consistency_loss = mesh_normal_consistency(pred_lv_mesh)
    hippo_normal_consistency_loss = mesh_normal_consistency(pred_hippo_mesh)
    
    
    loss =  lda[0]*  (lv_point_mesh_dist_loss + hippo_point_mesh_dist_loss)\
    + lda[1] * (lv_chamfer_loss + hippo_chamfer_loss)\
    + lda[2] * (lv_edge_loss + hippo_edge_loss)\
    + lda[3] * (lv_laplacian_loss+hippo_laplacian_loss)\
    + lda[4]* (lv_normal_consistency_loss+hippo_normal_consistency_loss)\

    log = {"loss": loss,
        "point_mesh_dist_loss": lv_point_mesh_dist_loss.detach()+ hippo_point_mesh_dist_loss.detach(),
        "chamfer_loss": lv_chamfer_loss.detach()+lv_chamfer_loss.detach(),
        "edge_loss": lv_edge_loss.detach()+hippo_edge_loss.detach(),
        "laplacian_loss": lv_laplacian_loss.detach()+hippo_laplacian_loss.detach(),
        "normal_consistency_loss": lv_normal_consistency_loss.detach()+hippo_normal_consistency_loss.detach()
        }
    
    return loss, log

def lv_hippo_criterion2(verts, lv_target, hippo_target, lv_tri, hippo_tri, lda=[2,2,2000,100,500,1,1]):
    loss=0.0

    # cf loss
    pred_lv_mesh = Meshes(verts=list(verts), faces=list(lv_tri))
    pred_lv_points = sample_points_from_meshes(pred_lv_mesh, lv_target.shape[1])
    lv_chamfer_loss =  chamfer_distance(pred_lv_points, lv_target)[0]
    pred_hippo_mesh = Meshes(verts=list(verts), faces=list(hippo_tri))
    pred_hippo_points = sample_points_from_meshes(pred_hippo_mesh, hippo_target.shape[1])
    hippo_chamfer_loss =  chamfer_distance(pred_hippo_points, hippo_target)[0]
    
    # point-mesh loss
    lv_pointclouds = Pointclouds(points=lv_target)
    lv_point_mesh_dist_loss = point_mesh_face_distance2(pred_lv_mesh, lv_pointclouds)
    hippo_pointclouds = Pointclouds(points=hippo_target)
    hippo_point_mesh_dist_loss = point_mesh_face_distance2(pred_hippo_mesh, hippo_pointclouds)

    # Regularization loss
    lv_edge_loss = mesh_edge_var_loss(pred_lv_mesh)
    hippo_edge_loss = mesh_edge_var_loss(pred_hippo_mesh)
    lv_laplacian_loss =  mesh_laplacian_smoothing(pred_lv_mesh, method="cotcurv")
    hippo_laplacian_loss =  mesh_laplacian_smoothing(pred_hippo_mesh, method="cotcurv")
    lv_normal_consistency_loss = mesh_normal_consistency(pred_lv_mesh)
    hippo_normal_consistency_loss = mesh_normal_consistency(pred_hippo_mesh)
    
    
    loss =  lda[0]*  (lv_point_mesh_dist_loss + hippo_point_mesh_dist_loss)\
    + lda[1] * (lv_chamfer_loss + hippo_chamfer_loss)\
    + lda[2] * (lv_edge_loss + hippo_edge_loss)\
    + lda[3] * (lv_laplacian_loss+hippo_laplacian_loss)\
    + lda[4]* (lv_normal_consistency_loss+hippo_normal_consistency_loss)\

    log = {"loss": loss,
        "point_mesh_dist_loss": lv_point_mesh_dist_loss.detach()+ hippo_point_mesh_dist_loss.detach(),
        "chamfer_loss": lv_chamfer_loss.detach()+lv_chamfer_loss.detach(),
        "edge_loss": lv_edge_loss.detach()+hippo_edge_loss.detach(),
        "laplacian_loss": lv_laplacian_loss.detach()+hippo_laplacian_loss.detach()*0.1,
        "normal_consistency_loss": lv_normal_consistency_loss.detach()+hippo_normal_consistency_loss.detach()
        }
    
    return loss, log




def temp_criterion(verts, target, faces):
    def relu(value):
        if value<0: return 0
        else: return value
    loss=0.0

    # cf loss
    pred_mesh = Meshes(verts=list(verts), faces=list(faces))
    pred_points = sample_points_from_meshes(pred_mesh, target.shape[1])
    chamfer_loss =  chamfer_distance(pred_points, target)[0]
    
    # point-mesh loss
    pointclouds = Pointclouds(points=target)
    point_mesh_dist_loss = point_mesh_face_distance(pred_mesh, pointclouds)

    # Regularization loss
    edge_loss = mesh_edge_var_loss(pred_mesh)
    laplacian_loss =  mesh_laplacian_smoothing(pred_mesh, method="uniform")
    normal_consistency_loss = mesh_normal_consistency(pred_mesh)
    edge_loss = mesh_edge_var_loss(pred_mesh)
    
    loss =   0.5*(point_mesh_dist_loss)\
    + 0.0 * (chamfer_loss)\
    + 1500 * (edge_loss)\
    + 1 * (laplacian_loss)\
    + 0.5 * (normal_consistency_loss)\

    log = {"loss": loss,
        "point_mesh_dist_loss": point_mesh_dist_loss.detach(),
        "chamfer_loss": chamfer_loss.detach(),
        "edge_loss": edge_loss.detach(),
        "laplacian_loss": laplacian_loss.detach(),
        "normal_consistency_loss": normal_consistency_loss.detach()
        }
    
    return loss, log

def tex_loss(bf_vert, vert_tex, target_tex)-> torch.Tensor:
    
    bf_tex = torch.zeros(bf_vert.shape).cuda()
    print(f"{bf_vert.requires_grad=}, {bf_tex.requires_grad=}")
    
    for n in range(bf_vert.shape[0]):
        a= []
        pt_x = int(bf_vert[n,0])
        pt_y = int(bf_vert[n,1])
        pt_z = int(bf_vert[n,2])

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    tex = target_tex[i+pt_x,j+pt_y,k+pt_z]
                    a.append(tex)
        counts = Counter(a)
        most_common_value = max(counts, key=lambda x: counts[x] if (x != 4 and x!=5 and x!=31 and x!=0) else -1)
        #if most_common_value == 4: print("4 is here!")
        bf_tex[n] = int(most_common_value)
    
    accuracy = torch.sum(vert_tex[:, 0] == bf_tex[:,0])/bf_tex.shape[0]


    return accuracy
    return loss, log
def lvhippo_loss(verts, lv_faces, hippo_faces, lv_target, hippo_target, epoch):
    # hippo_target = target[3]
    lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_faces))
    hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
    # print(f"{torch.sum(lv_pred_mesh.verts_normals_packed()!=0.0)=}")
    chamfer_losses=[]
    
    lv_laplacian_loss = mesh_laplacian_smoothing(lv_pred_mesh, method="uniform")
    hippo_laplacian_loss = mesh_laplacian_smoothing(hippo_pred_mesh, method="uniform")

    lv_normal_consistency_loss = mesh_normal_consistency(lv_pred_mesh)
    hippo_normal_consistency_loss = mesh_normal_consistency(hippo_pred_mesh)

    lv_edge_loss = mesh_edge_var_loss(lv_pred_mesh)
    hippo_edge_loss = mesh_edge_var_loss(hippo_pred_mesh)

    lv_pcd = Pointclouds(points=lv_target)
    hippo_pcd = Pointclouds(points=hippo_target)

    lv_point_mesh_dist_loss = point_mesh_face_distance(lv_pred_mesh, lv_pcd)
    hippo_point_mesh_dist_loss = point_mesh_face_distance(hippo_pred_mesh, hippo_pcd)

    loss =     3000* (lv_edge_loss+hippo_edge_loss)\
    + (15*lv_normal_consistency_loss + 3*hippo_normal_consistency_loss)\
    + (lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss)*1.5\
    +1* (lv_laplacian_loss+hippo_laplacian_loss-38.0)
    cf_loss =0.0
    cf_log = {}

    
    lv_chamfer_num=2000
    lv_pred_points = sample_points_from_meshes(lv_pred_mesh, lv_chamfer_num)
    lv_random_indices = torch.randperm(lv_target.size(1))[:lv_chamfer_num]
    lv_target_points = lv_target[:, lv_random_indices, :]
    lv_chamfer_loss, _ = chamfer_distance(lv_pred_points, lv_target_points)
    
    hippo_chamfer_num=200
    hippo_pred_points = sample_points_from_meshes(hippo_pred_mesh, hippo_chamfer_num)
    hippo_random_indices = torch.randperm(hippo_target.size(1))[:hippo_chamfer_num]
    hippo_target_points = hippo_target[:,hippo_random_indices, :]
    hippo_chamfer_loss, _ = chamfer_distance(hippo_pred_points, hippo_target_points)
    
    loss += (lv_chamfer_loss+hippo_chamfer_loss)*4.5
        
    # if epoch<5000: loss = lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss
        
    log = {"loss": loss.detach(),
           "chamfer_loss": lv_chamfer_loss.detach()+hippo_chamfer_loss.detach(), 
           "normal_consistency_loss": lv_normal_consistency_loss.detach()+hippo_normal_consistency_loss.detach(),
           "edge_loss": lv_edge_loss.detach()+hippo_edge_loss.detach(),
           "laplacian_loss": lv_laplacian_loss.detach()+hippo_laplacian_loss.detach()-180,
           "point_mesh_dist_loss": lv_point_mesh_dist_loss.detach()+hippo_point_mesh_dist_loss.detach()
          }
#     loss += lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss

    print(loss)
    return loss, log


def lvhippo_texloss(verts, lv_faces, hippo_faces, tri, target, target_lv, epoch=0):
    ### Regularization loss
    lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_faces))
    hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
        
    lv_laplacian_loss = mesh_laplacian_smoothing(lv_pred_mesh, method="uniform")
    hippo_laplacian_loss = mesh_laplacian_smoothing(hippo_pred_mesh, method="uniform")

    lv_normal_consistency_loss = mesh_normal_consistency(lv_pred_mesh)
    hippo_normal_consistency_loss = mesh_normal_consistency(hippo_pred_mesh)

    lv_edge_loss = mesh_edge_var_loss(lv_pred_mesh)
    hippo_edge_loss = mesh_edge_var_loss(hippo_pred_mesh)

    ### Distance loss

    # Cf loss
    lv_chamfer_num=2000
    lv_pred_points = sample_points_from_meshes(lv_pred_mesh, lv_chamfer_num)
    lv_random_indices = torch.randperm(target_lv.size(1))[:lv_chamfer_num]
    lv_target_points = target_lv[:, lv_random_indices, :]
    lv_chamfer_loss, _ = chamfer_distance(lv_pred_points, lv_target_points)
    
    hippo_chamfer_num=200
    hippo_pred_points = sample_points_from_meshes(hippo_pred_mesh, hippo_chamfer_num)
    hippo_random_indices = torch.randperm(target[3].size(1))[:hippo_chamfer_num]
    hippo_target_points = target[3][:,hippo_random_indices, :]
    hippo_chamfer_loss, _ = chamfer_distance(hippo_pred_points, hippo_target_points)
   
    # PM loss   
    lv_pcd = Pointclouds(points=target_lv)
    hippo_pcd = Pointclouds(points=target[3])

    lv_point_mesh_dist_loss = point_mesh_face_distance2(lv_pred_mesh, lv_pcd)
    hippo_point_mesh_dist_loss = point_mesh_face_distance2(hippo_pred_mesh, hippo_pcd)

    point_mesh_dist_loss = 0
    chamfer_lv_hippo_loss = (lv_chamfer_loss+hippo_chamfer_loss)
    chamfer_loss = 0
    all_sample = 0
    all_mesh =0
    # for t in target:
    #     all_sample += t.size(1)
    # for face in tri:
    #     all_mesh += face.size(1) 
    for index in [0,1,2,3,4]:
        # pm loss
        face = tri[index]
        target_points = target[index]

        pred_mesh = Meshes(verts=list(verts), faces=list(face))
        pointclouds = Pointclouds(points=target_points)
        pm_loss = point_mesh_face_distance(pred_mesh, pointclouds)
        
        
        # cf losss
        chamfer_num = 100
        pred_points = sample_points_from_meshes(pred_mesh, chamfer_num)
        random_indices = torch.randperm(target_points.size(1))[:chamfer_num]
        target_points = target_points[:, random_indices, :]
        cf_loss, _ = chamfer_distance(pred_points, target_points)
        if index ==4:
            chamfer_loss += cf_loss*1.2
            point_mesh_dist_loss += pm_loss*1.2
        else:
            chamfer_loss += cf_loss
            point_mesh_dist_loss += pm_loss
    
    point_mesh_lv_hippo = lv_point_mesh_dist_loss + hippo_point_mesh_dist_loss

    loss =  4700* (lv_edge_loss+hippo_edge_loss)\
    + ((18*lv_normal_consistency_loss + 3*hippo_normal_consistency_loss))\
    + 1.7* (lv_laplacian_loss+hippo_laplacian_loss-38.0)\
    + (point_mesh_lv_hippo) * 1.0\
    + chamfer_lv_hippo_loss * 3.0\
    + point_mesh_dist_loss * 1.5 * 1.5\
    + (chamfer_loss) * 1.5 * 3.0

        
    # if epoch<5000: loss = lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss

        
    log = {"loss": loss.detach(),
           "chamfer_loss": chamfer_lv_hippo_loss.detach(), 
           "chamfer_loss_tex": chamfer_loss.detach(), 
           "normal_consistency_loss": lv_normal_consistency_loss.detach()+hippo_normal_consistency_loss.detach(),
           "edge_loss": lv_edge_loss.detach()+hippo_edge_loss.detach(),
           "laplacian_loss": lv_laplacian_loss.detach()+hippo_laplacian_loss.detach()-180,
           "point_mesh_dist_loss": point_mesh_lv_hippo.detach(),
           "point_mesh_dist_loss_tex": point_mesh_dist_loss.detach()
          }
#     loss += lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss

    print(loss)
    return loss, log


def onlylv_loss(verts, lv_faces, hippo_faces, lv_target, hippo_target):
    lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_faces))
    hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
    
    chamfer_losses=[]
    
    lv_laplacian_loss = mesh_laplacian_smoothing(lv_pred_mesh, method="uniform")
    hippo_laplacian_loss = 0

    lv_normal_consistency_loss = mesh_normal_consistency(lv_pred_mesh)
    hippo_normal_consistency_loss = 0

    
    #lv_edge_loss = mesh_edge_loss(lv_pred_mesh,2.8)
    #hippo_edge_loss = mesh_edge_loss(hippo_pred_mesh,2.8)
    lv_edge_loss = mesh_edge_var_loss(lv_pred_mesh)
    hippo_edge_loss =0

    lv_pcd = Pointclouds(points=lv_target)
    hippo_pcd = Pointclouds(points=hippo_target)

    lv_point_mesh_dist_loss = point_mesh_face_distance2(lv_pred_mesh, lv_pcd)
    hippo_point_mesh_dist_loss = 0
    
    #normal_loss = normal_L2_loss(hippo_pred_mesh)
    loss =  5* (lv_laplacian_loss)\
    + 1500* (lv_edge_loss)\
    + (lv_normal_consistency_loss)\
    + (lv_point_mesh_dist_loss)*3
    cf_loss =0.0
    cf_log = {}
            
    lv_chamfer_num=2000
    lv_pred_points = sample_points_from_meshes(lv_pred_mesh, lv_chamfer_num)
    lv_random_indices = torch.randperm(lv_target.size(1))[:lv_chamfer_num]
    lv_target_points = lv_target[:, lv_random_indices, :]
    lv_chamfer_loss, _ = chamfer_distance(lv_pred_points, lv_target_points)
    
    hippo_chamfer_num=200
    hippo_pred_points = sample_points_from_meshes(hippo_pred_mesh, hippo_chamfer_num)
    hippo_random_indices = torch.randperm(hippo_target.size(1))[:hippo_chamfer_num]
    hippo_target_points = hippo_target[:,hippo_random_indices, :]
    hippo_chamfer_loss, _ = chamfer_distance(hippo_pred_points, hippo_target_points)
    hippo_chamfer_loss =0
    
    loss += (lv_chamfer_loss)*0.5
        
    
    log = {"loss": loss.detach(),
           "chamfer_loss": lv_chamfer_loss.detach(), 
           "normal_consistency_loss": lv_normal_consistency_loss.detach(),
           "edge_loss": lv_edge_loss.detach(),
           "laplacian_loss": lv_laplacian_loss.detach(),
           "point_mesh_dist_loss": lv_point_mesh_dist_loss.detach()
          }
#     loss = lv_point_mesh_dist_loss

    print(loss)
    return loss, log

def onlyhippo_loss(verts, lv_faces, hippo_faces, lv_target, hippo_target):
    lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_faces))
    hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
    
    chamfer_losses=[]
    
    lv_laplacian_loss = 0
    hippo_laplacian_loss = mesh_laplacian_smoothing(hippo_pred_mesh, method="uniform")

    lv_normal_consistency_loss = 0
    hippo_normal_consistency_loss = mesh_normal_consistency(hippo_pred_mesh)

    
    #lv_edge_loss = mesh_edge_loss(lv_pred_mesh,2.8)
    #hippo_edge_loss = mesh_edge_loss(hippo_pred_mesh,2.8)
    lv_edge_loss = 0
    hippo_edge_loss =mesh_edge_var_loss(hippo_pred_mesh)

    lv_pcd = Pointclouds(points=lv_target)
    hippo_pcd = Pointclouds(points=hippo_target)

    lv_point_mesh_dist_loss = 0
    hippo_point_mesh_dist_loss = point_mesh_face_distance2(hippo_pred_mesh, hippo_pcd)
    
    #normal_loss = normal_L2_loss(hippo_pred_mesh)
    loss =  5* (hippo_laplacian_loss)\
    + 1500* (hippo_edge_loss)\
    + (hippo_normal_consistency_loss)\
    + (hippo_point_mesh_dist_loss)*3
    cf_loss =0.0
    cf_log = {}


    lv_chamfer_num=2000
    lv_pred_points = sample_points_from_meshes(lv_pred_mesh, lv_chamfer_num)
    lv_random_indices = torch.randperm(lv_target.size(1))[:lv_chamfer_num]
    lv_target_points = lv_target[:, lv_random_indices, :]
    lv_chamfer_loss, _ = chamfer_distance(lv_pred_points, lv_target_points)
    lv_chamfer_loss =0
    hippo_chamfer_num=200
    hippo_pred_points = sample_points_from_meshes(hippo_pred_mesh, hippo_chamfer_num)
    hippo_random_indices = torch.randperm(hippo_target.size(1))[:hippo_chamfer_num]
    hippo_target_points = hippo_target[:,hippo_random_indices, :]
    hippo_chamfer_loss, _ = chamfer_distance(hippo_pred_points, hippo_target_points)
    
    
    loss += (hippo_chamfer_loss)*0.5
        
    
    log = {"loss": loss.detach(),
           "chamfer_loss": hippo_chamfer_loss.detach(), 
           "normal_consistency_loss": hippo_normal_consistency_loss.detach(),
           "edge_loss": hippo_edge_loss.detach(),
           "laplacian_loss": hippo_laplacian_loss.detach(),
           "point_mesh_dist_loss": hippo_point_mesh_dist_loss.detach()
          }
#     loss = lv_point_mesh_dist_loss

    print(loss)
    return loss, log


def lvhippo_texloss_r(verts, lv_faces, hippo_faces, tri, target, target_lv, epoch=0):
    ### Regularization loss
    lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_faces))
    hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
        
    lv_laplacian_loss = mesh_laplacian_smoothing(lv_pred_mesh, method="uniform")
    hippo_laplacian_loss = mesh_laplacian_smoothing(hippo_pred_mesh, method="uniform")

    lv_normal_consistency_loss = mesh_normal_consistency(lv_pred_mesh)
    hippo_normal_consistency_loss = mesh_normal_consistency(hippo_pred_mesh)

    lv_edge_loss = mesh_edge_var_loss(lv_pred_mesh)
    hippo_edge_loss = mesh_edge_var_loss(hippo_pred_mesh)

    ### Distance loss

    # Cf loss
    lv_chamfer_num=2000
    lv_pred_points = sample_points_from_meshes(lv_pred_mesh, lv_chamfer_num)
    lv_random_indices = torch.randperm(target_lv.size(1))[:lv_chamfer_num]
    lv_target_points = target_lv[:, lv_random_indices, :]
    lv_chamfer_loss, _ = chamfer_distance(lv_pred_points, lv_target_points)
    
    hippo_chamfer_num=200
    hippo_pred_points = sample_points_from_meshes(hippo_pred_mesh, hippo_chamfer_num)
    hippo_random_indices = torch.randperm(target[3].size(1))[:hippo_chamfer_num]
    hippo_target_points = target[3][:,hippo_random_indices, :]
    hippo_chamfer_loss, _ = chamfer_distance(hippo_pred_points, hippo_target_points)
   
    # PM loss   
    lv_pcd = Pointclouds(points=target_lv)
    hippo_pcd = Pointclouds(points=target[3])

    lv_point_mesh_dist_loss = point_mesh_face_distance2(lv_pred_mesh, lv_pcd)
    hippo_point_mesh_dist_loss = point_mesh_face_distance2(hippo_pred_mesh, hippo_pcd)

    point_mesh_dist_loss = 0
    chamfer_lv_hippo_loss = (lv_chamfer_loss+hippo_chamfer_loss)
    chamfer_loss = 0
    all_sample = 0
    all_mesh =0
    # for t in target:
    #     all_sample += t.size(1)
    # for face in tri:
    #     all_mesh += face.size(1) 
    for index in [0,1,2,3,4]:
        # pm loss
        face = tri[index]
        target_points = target[index]

        pred_mesh = Meshes(verts=list(verts), faces=list(face))
        pointclouds = Pointclouds(points=target_points)
        pm_loss = point_mesh_face_distance(pred_mesh, pointclouds)
        point_mesh_dist_loss += pm_loss
        
        # cf losss
        chamfer_num = 100
        pred_points = sample_points_from_meshes(pred_mesh, chamfer_num)
        random_indices = torch.randperm(target_points.size(1))[:chamfer_num]
        target_points = target_points[:, random_indices, :]
        cf_loss, _ = chamfer_distance(pred_points, target_points)
        chamfer_loss += cf_loss
    
    point_mesh_lv_hippo = lv_point_mesh_dist_loss + hippo_point_mesh_dist_loss

    loss =  4000* (lv_edge_loss+hippo_edge_loss)\
    + ((15*lv_normal_consistency_loss + 3*hippo_normal_consistency_loss))\
    + 1* (lv_laplacian_loss+hippo_laplacian_loss-38.0)\
    + (point_mesh_lv_hippo) * 1.0\
    + chamfer_lv_hippo_loss * 3.0\
    + (chamfer_loss) * 2.0 * 3.0

        
    # if epoch<5000: loss = lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss

        
    log = {"loss": loss.detach(),
           "chamfer_loss": chamfer_lv_hippo_loss.detach(), 
           "chamfer_loss_tex": chamfer_loss.detach(), 
           "normal_consistency_loss": lv_normal_consistency_loss.detach()+hippo_normal_consistency_loss.detach(),
           "edge_loss": lv_edge_loss.detach()+hippo_edge_loss.detach(),
           "laplacian_loss": lv_laplacian_loss.detach()+hippo_laplacian_loss.detach()-180,
           "point_mesh_dist_loss": point_mesh_lv_hippo.detach(),
           "point_mesh_dist_loss_tex": point_mesh_dist_loss.detach()
          }
#     loss += lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss

    print(loss)
    return loss, log

def lv_hippo_tex_criterion2(verts, lv_target, hippo_target,tex1_target, tex2_target, tex3_target, lv_tri, hippo_tri, tri1, tri2, tri3, lda=[2, 2, 2000, 100, 500, 1, 1]):
    loss=0.0

    # cf loss
    pred_lv_mesh = Meshes(verts=list(verts), faces=list(lv_tri))
    pred_lv_points = sample_points_from_meshes(pred_lv_mesh, lv_target.shape[1])
    lv_chamfer_loss =  chamfer_distance(pred_lv_points, lv_target)[0]
    pred_hippo_mesh = Meshes(verts=list(verts), faces=list(hippo_tri)) 
    pred_hippo_points = sample_points_from_meshes(pred_hippo_mesh, hippo_target.shape[1])
    hippo_chamfer_loss =  chamfer_distance(pred_hippo_points, hippo_target)[0]
    
    pred_tri1_mesh = Meshes(verts=list(verts), faces=list(tri1))
    pred_tri2_mesh = Meshes(verts=list(verts), faces=list(tri2))
    pred_tri3_mesh = Meshes(verts=list(verts), faces=list(tri3))
    
    # tex1_pointclouds = Pointclouds(points=tex1_target)
    # tex2_pointclouds = Pointclouds(points=tex2_target)
    # tex3_pointclouds = Pointclouds(points=tex3_target)
    # tex1_point_mesh_dist_loss = point_mesh_face_distance(pred_tri1_mesh, tex1_pointclouds)
    # tex2_point_mesh_dist_loss = point_mesh_face_distance(pred_tri2_mesh, tex2_pointclouds)
    # tex3_point_mesh_dist_loss = point_mesh_face_distance(pred_tri3_mesh, tex3_pointclouds)

    pred_tri1_points = sample_points_from_meshes(pred_tri1_mesh, tex1_target.shape[1])
    pred_tri2_points = sample_points_from_meshes(pred_tri2_mesh, tex2_target.shape[1])
    pred_tri3_points = sample_points_from_meshes(pred_tri3_mesh, tex3_target.shape[1])
    
    tri1_chamfer_loss =  chamfer_distance(pred_tri1_points, tex1_target)[0]
    tri2_chamfer_loss =  chamfer_distance(pred_tri2_points, tex2_target)[0]
    tri3_chamfer_loss =  chamfer_distance(pred_tri3_points, tex3_target)[0]
    
    # point-mesh loss
    lv_pointclouds = Pointclouds(points=lv_target)
    lv_point_mesh_dist_loss = point_mesh_face_distance(pred_lv_mesh, lv_pointclouds)
    hippo_pointclouds = Pointclouds(points=hippo_target)
    hippo_point_mesh_dist_loss = point_mesh_face_distance(pred_hippo_mesh, hippo_pointclouds)

    # Regularization loss
    lv_edge_loss = mesh_edge_var_loss(pred_lv_mesh)
    hippo_edge_loss = mesh_edge_var_loss(pred_hippo_mesh)
    lv_laplacian_loss =  mesh_laplacian_smoothing(pred_lv_mesh, method="cotcurv")
    hippo_laplacian_loss =  mesh_laplacian_smoothing(pred_hippo_mesh, method="cotcurv")
    lv_normal_consistency_loss = mesh_normal_consistency(pred_lv_mesh)
    hippo_normal_consistency_loss = mesh_normal_consistency(pred_hippo_mesh)
    
    
    loss =  lda[0]*  (lv_point_mesh_dist_loss + hippo_point_mesh_dist_loss)\
    + lda[1] * (lv_chamfer_loss + hippo_chamfer_loss)\
    + lda[2] * (lv_edge_loss + hippo_edge_loss)\
    + lda[3] * (lv_laplacian_loss+hippo_laplacian_loss)\
    + lda[4] * (lv_normal_consistency_loss+hippo_normal_consistency_loss)\
    + lda[1] * (tri1_chamfer_loss + tri2_chamfer_loss + tri3_chamfer_loss)/2 # tex loss

    log = {"loss": loss,
        "point_mesh_dist_loss": lv_point_mesh_dist_loss.detach()+ hippo_point_mesh_dist_loss.detach(),
        "chamfer_loss": lv_chamfer_loss.detach()+lv_chamfer_loss.detach(),
        "tri_loss": tri1_chamfer_loss.detach()+tri2_chamfer_loss.detach()+tri3_chamfer_loss.detach(),
        "edge_loss": lv_edge_loss.detach()+hippo_edge_loss.detach(),
        "laplacian_loss": lv_laplacian_loss.detach()+hippo_laplacian_loss.detach(),
        "normal_consistency_loss": lv_normal_consistency_loss.detach()+hippo_normal_consistency_loss.detach()
        }
    
    return loss, log, lv_chamfer_loss