
"""
Modified code from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master
"""
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle

import datetime
import logging
import importlib
import shutil
import argparse
import glob 
from pathlib import Path
from tqdm import tqdm
from models.loss import lv_hippo_tex_criterion2
from torch.utils.data import DataLoader
from pytorch3d.loss import mesh_laplacian_smoothing
from models.pointnet import PointNetOpt
from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.tensorboard import SummaryWriter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--epoch', default=5001, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--data_path', type=str, default='../data/LV_left/train_data.pickle', help='data path to optimize')
    parser.add_argument('--tag', default='manual', help='tex or point')
    parser.add_argument('--sub_id', default= '', help='LBC subject id')
    parser.add_argument('--lda', type=float, nargs='+', default= [3,0.5,1500,5,1,1,1], help='LBC subject id')

    return parser.parse_args()

def train_data(data_file, tag=None, id=None):
    temp_data_file = "../temp_meshes/cuttail_temp_L_tex.pkl"
    
    with open(temp_data_file, "rb") as f:
        temp_data = pickle.load(f)
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    print("@@@@@!!",id)
 
 
    # Convert vertices and faces to tensors and move to GPU
    vert = np.asarray(data['vert']).astype(np.float32)
    lv_tri = np.asarray(temp_data['cuttail_lv']).astype(np.float32)
    hippo_tri = np.asarray(temp_data['cuttail_hippo']).astype(np.float32)
    tri1 = np.asarray(temp_data['tri1']).astype(np.float32)#thalamus
    tri2 = np.asarray(temp_data['tri2']).astype(np.float32)#caudate
    tri3 = np.asarray(temp_data['tri3']).astype(np.float32)#oppositeLV
    lv_pt = np.asarray(data['lv']).astype(np.float32)
    hippo_pt = np.asarray(data['hippo']).astype(np.float32)
    tex1_pt = np.asarray(data['tex1']).astype(np.float32)
    tex2_pt = np.asarray(data['tex2']).astype(np.float32)
    tex3_pt = np.asarray(data['tex3']).astype(np.float32)

    # # move to origin

    # hippo_pt[:,0] -= lv_pt[:,0].min()
    # vert[:,0] = vert[:,0] - vert[:,0].min()
    # lv_pt[:,0] -= lv_pt[:,0].min()
    
    # hippo_pt[:,1] -= lv_pt[:,1].max()
    # vert[:,1] = vert[:,1] - vert[:,1].max()
    # lv_pt[:,1] -= lv_pt[:,1].max()
    
    # tgt_vert= np.concatenate((lv_pt, hippo_pt), axis=0)
    # hippo_pt[:,2] = hippo_pt[:,2] - (tgt_vert[:,2].max()+tgt_vert[:,2].min())/2
    # lv_pt[:,2] = lv_pt[:,2] - (tgt_vert[:,2].max()+tgt_vert[:,2].min())/2
    # vert[:,2] = vert[:,2] - (vert[:,2].max()+vert[:,2].min())/2
    
    # vert = vert * scaler[0,0,:]
    vertices = torch.from_numpy(vert[np.newaxis, :, :]).cuda()
    lv_tri = torch.from_numpy(lv_tri[np.newaxis, :, :]).cuda()
    hippo_tri = torch.from_numpy(hippo_tri[np.newaxis, :, :]).cuda()
    tri1 = torch.from_numpy(tri1[np.newaxis, :, :]).cuda()
    tri2 = torch.from_numpy(tri2[np.newaxis, :, :]).cuda()
    tri3 = torch.from_numpy(tri3[np.newaxis, :, :]).cuda()


    lv_target = torch.from_numpy(lv_pt[np.newaxis, :, :]).cuda()
    hippo_target = torch.from_numpy(hippo_pt[np.newaxis, :, :]).cuda()
    
    tex1_target = torch.from_numpy(tex1_pt[np.newaxis, :, :]).cuda()
    tex2_target = torch.from_numpy(tex2_pt[np.newaxis, :, :]).cuda()
    tex3_target = torch.from_numpy(tex3_pt[np.newaxis, :, :]).cuda()
    
    lv_pred_mesh = Meshes(verts=list(vertices), faces=list(lv_tri))
    hippo_pred_mesh = Meshes(verts=list(vertices), faces=list(hippo_tri))   
    
    print(mesh_laplacian_smoothing(lv_pred_mesh), mesh_laplacian_smoothing(hippo_pred_mesh), "BASIC LOSS")
    return vertices, lv_tri, hippo_tri,  lv_target, hippo_target, tex1_target, tex2_target, tex3_target, lv_pred_mesh, hippo_pred_mesh, tri1, tri2, tri3
def create_directory(directory_path, log=False):
    if not os.path.exists(directory_path):
        # Create target Directory
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else: 
        print(f"{directory_path} already exists")

def main(args):
    '''LOG'''
    args = parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    batch_size = args.batch_size
    print(args.sub_id,"!!!!!")

    create_directory(f"../optim_result/{args.tag}/log/{args.sub_id}", True)
    writer = SummaryWriter(log_dir=f"../optim_result/{args.tag}/log/{args.sub_id}")
    
    '''MODEL LOADING'''
    model = PointNetOpt(num_classes=3, input_transform=False, feature_transform=False).to("cuda")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch//5, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    global_step = 0
    '''TRANING'''
    print('Start training...')
    model.train()
    # data preparation
    vertices, lv_tri, hippo_tri,  lv_target, hippo_target, tex1_target, tex2_target, tex3_target, lv_pred_mesh, hippo_pred_mesh, tri1, tri2, tri3 = train_data(args.data_path, args.tag, id = args.sub_id)
    # Convert vertices and faces to tensors and move to GPU
    orig_vertices = vertices.clone()
    lv_orig_norm = lv_pred_mesh.verts_normals_packed().clone()
    hippo_orig_norm = hippo_pred_mesh.verts_normals_packed().clone()

    print(f"lambda is following. \n 1. pm: {args.lda[0]}\n 2. cf {args.lda[1]}\n 3. edge {args.lda[2]}\n 4. lap {args.lda[3]}\n 5. norm_con {args.lda[4]}\n 6. l2_vert {args.lda[5]}\n 7.l2_norm {args.lda[6]}")
    # Optimization loop
    loss_min = 10000
    saved_vert = orig_vertices
    lda = args.lda
    for epoch in tqdm(range(args.epoch)):

        optimizer.zero_grad()

        vertices = lv_pred_mesh.verts_list()[0].unsqueeze(0)

        # Forward pass
        pred = model(vertices)
        
        # Avoid inplace operations by creating new tensors
        verts = vertices.clone() + pred.transpose(2, 1).clone()
        l2_loss = loss_fn(orig_vertices, verts)
        if epoch == 1000: 
            lda = [2, 2, 200, 10, 50, 1, 1]
            print(f"lda changed into: {lda}")
            
        loss, log, lv_chamfer_loss = lv_hippo_tex_criterion2(verts, lv_target, hippo_target,tex1_target, tex2_target, tex3_target, lv_tri, hippo_tri, tri1, tri2, tri3, lda)
        if epoch%1000 == 0 and epoch>=5000 and lv_chamfer_loss > 1:
            args.epoch+=1000
            print(lv_chamfer_loss) 
        loss += l2_loss * args.lda[-2]
        
        mid_lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_tri))
        mid_lv_norm = mid_lv_pred_mesh.verts_normals_packed()
        l2_lv_norm_loss = loss_fn(lv_orig_norm, mid_lv_norm)
        mid_hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_tri))
        mid_hippo_norm = mid_hippo_pred_mesh.verts_normals_packed()
        l2_hippo_norm_loss = loss_fn(hippo_orig_norm, mid_hippo_norm)
        loss += (l2_hippo_norm_loss+l2_lv_norm_loss) * args.lda[-1]

        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        global_step += 1
        log['l2 loss'] = l2_loss
        log['l2 lv norm loss'] = l2_lv_norm_loss
        log['l2 hippo norm loss'] = l2_hippo_norm_loss

        if loss < loss_min:
            loss_min = loss
            saved_vert = verts

        if epoch % 100 == 0 : print(epoch, log)
        
        for key in log.keys():
            writer.add_scalar(key, log[key], global_step=epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch)
        if epoch % 500 == 0 or epoch == args.epoch-1:
            verts_np = verts.detach().cpu().numpy()
            create_directory(rf'../optim_result/{args.tag}/out/{args.sub_id}')
            np.save(rf'../optim_result/{args.tag}/out/{args.sub_id}/{epoch}_{args.sub_id}_{loss}.npy', verts_np)
    np.save(rf'../optim_result/{args.tag}/out/{args.sub_id}/smallest_{args.sub_id}_{loss_min}.npy', saved_vert.detach().cpu().numpy())
    print('End of training...')
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
