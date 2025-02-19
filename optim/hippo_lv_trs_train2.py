
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
from models.loss import lv_hippo_criterion
from torch.utils.data import DataLoader

from models.pointnet import PointNetOpt, PointNetCls
from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)  # Add this line


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--epoch', default=2001, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--data_path', type=str, default='../data/LV_left/train_data.pickle', help='data path to optimize')
    parser.add_argument('--tag', default='manual', help='tex or point')
    parser.add_argument('--sub_id', default= '', help='LBC subject id')
    parser.add_argument('--lda', type=float, nargs='+', default= [3,0.5,1500,5,1,1,1], help='LBC subject id')

    return parser.parse_args()

def train_data(data_file, tag=None, id=None):
    temp_data_file = "/root/LV/LV/2502LV/cuttail_temp.pkl"
    
    with open(temp_data_file, "rb") as f:
        temp_data = pickle.load(f)
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    print("@@@@@!!",id)
    # print(glob.glob(f"/root/LV/lv-parametric-modelling/ipynb/MICCAI-LV/results/mid_TRS/out/{id}/2000_*.npy"))
    # scaler = np.load(glob.glob(f"/root/LV/lv-parametric-modelling/ipynb/MICCAI-LV/results/mid_TRS/out/{id}/2000_*.npy")[0])
    # print(scaler)
 
    # Convert vertices and faces to tensors and move to GPU
    vert = np.asarray(data['vert']).astype(np.float32)
    lv_tri = np.asarray(temp_data['lv']).astype(np.float32)
    hippo_tri = np.asarray(temp_data['hippo']).astype(np.float32)
    lv_pt = np.asarray(data['lv']).astype(np.float32)
    hippo_pt = np.asarray(data['hippo']).astype(np.float32)

    hippo_pt[:,0] -= lv_pt[:,0].min()
    vert[:,0] = vert[:,0] - vert[:,0].min()
    lv_pt[:,0] -= lv_pt[:,0].min()
    
    hippo_pt[:,1] -= lv_pt[:,1].max()
    vert[:,1] = vert[:,1] - vert[:,1].max()
    lv_pt[:,1] -= lv_pt[:,1].max()
    
    tgt_vert= np.concatenate((lv_pt, hippo_pt), axis=0)
    hippo_pt[:,2] = hippo_pt[:,2] - (tgt_vert[:,2].max()+tgt_vert[:,2].min())/2
    lv_pt[:,2] = lv_pt[:,2] - (tgt_vert[:,2].max()+tgt_vert[:,2].min())/2
    vert[:,2] = vert[:,2] - (vert[:,2].max()+vert[:,2].min())/2
    
    vertices = torch.from_numpy(vert[np.newaxis, :, :]).cuda()
    lv_tri = torch.from_numpy(lv_tri[np.newaxis, :, :]).cuda()
    hippo_tri = torch.from_numpy(hippo_tri[np.newaxis, :, :]).cuda()

    lv_target = torch.from_numpy(lv_pt[np.newaxis, :, :]).cuda()
    hippo_target = torch.from_numpy(hippo_pt[np.newaxis, :, :]).cuda()
    
    lv_pred_mesh = Meshes(verts=list(vertices), faces=list(lv_tri))
    hippo_pred_mesh = Meshes(verts=list(vertices), faces=list(hippo_tri))   

    return vertices, lv_tri, hippo_tri,  lv_target,hippo_target,lv_pred_mesh,hippo_pred_mesh
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

    create_directory(f"/root/LV/lv-parametric-modelling/ipynb/MICCAI-LV/results/{args.tag}/log/{args.sub_id}", True)
    writer = SummaryWriter(log_dir=f"/root/LV/lv-parametric-modelling/ipynb/MICCAI-LV/results/{args.tag}/log/{args.sub_id}")
    
    '''MODEL LOADING'''
    

    
    # data preparation
    vertices, lv_tri, hippo_tri, lv_target,hippo_target,lv_pred_mesh,hippo_pred_mesh = train_data(args.data_path, args.tag, args.sub_id)
    # Convert vertices and faces to tensors and move to GPU
    orig_vertices = vertices.clone()
    lv_orig_norm = lv_pred_mesh.verts_normals_packed().clone()
    hippo_orig_norm = hippo_pred_mesh.verts_normals_packed().clone()
    lv_face_norm = lv_pred_mesh.faces_normals_packed().clone()
    print(lv_face_norm.shape, "!!", lv_orig_norm.shape)
    print(f"lambda is following. \n 1. pm: {args.lda[0]}\n 2. cf {args.lda[1]}\n 3. edge {args.lda[2]}\n 4. lap {args.lda[3]}\n 5. norm_con {args.lda[4]}\n 6. l2_vert {args.lda[5]}\n 7.l2_norm {args.lda[6]}")
    # Optimization loop
    saved = torch.tensor([3]) 
    min = 1000
    
    
    model = PointNetCls(num_classes=3, input_transform=False, feature_transform=False).to("cuda")
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
    
    for epoch in tqdm(range(args.epoch)):

        optimizer.zero_grad()

        vertices = lv_pred_mesh.verts_list()[0].unsqueeze(0)

        # Forward pass
        pred = model(vertices)
        print(vertices.shape, pred.shape, "!!") #torch.Size([1, 2826, 3]) torch.Size([1, 9])
        print(pred, vertices[0,0])
        verts = vertices.clone() * (pred.unsqueeze(1))
        # verts[0] += pred[0,3:6]
        # print(pred, verts[0,0])

        l2_loss = loss_fn(orig_vertices, verts)
        loss, log = lv_hippo_criterion(verts, lv_target, hippo_target, lv_tri, hippo_tri, args.lda)
        loss += l2_loss * args.lda[-2]
        mid_lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_tri))
        mid_lv_norm = mid_lv_pred_mesh.verts_normals_packed()
        l2_lv_norm_loss = loss_fn(lv_orig_norm, mid_lv_norm)
        mid_hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_tri))
        mid_hippo_norm = mid_hippo_pred_mesh.verts_normals_packed()
        l2_hippo_norm_loss = loss_fn(hippo_orig_norm, mid_hippo_norm)
        loss += (l2_hippo_norm_loss+l2_lv_norm_loss) * args.lda[-1]

        if loss< min:
            min = loss
            saved = pred.unsqueeze(1)
            # print(saved)
        
        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        global_step += 1
        log['l2 loss'] = l2_loss
        log['l2 lv norm loss'] = l2_lv_norm_loss
        log['l2 hippo norm loss'] = l2_hippo_norm_loss
        if epoch % 100 == 0 : print(epoch, log)
        
        for key in log.keys():
            writer.add_scalar(key, log[key], global_step=epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch)
        if epoch % 1000 == 0 or epoch == args.epoch-1:
            create_directory(rf'/root/LV/lv-parametric-modelling/ipynb/MICCAI-LV/results/{args.tag}/out/{args.sub_id}')
            np.save(rf'/root/LV/lv-parametric-modelling/ipynb/MICCAI-LV/results/{args.tag}/out/{args.sub_id}/{epoch}_{args.sub_id}_{loss}.npy', saved.detach().cpu().numpy())
            # verts_np = verts.detach().cpu().numpy()
            # np.save(rf'/root/LV/lv-parametric-modelling/ipynb/MICCAI-LV/results/{args.tag}/out/{args.sub_id}/{epoch}_{args.sub_id}_{loss}.npy', verts_np)

    print('End of training...')
    print(args.sub_id,":", saved, '!!!')
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
    