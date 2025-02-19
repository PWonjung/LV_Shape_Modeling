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

from pathlib import Path
from tqdm import tqdm
from models.loss import  lvhippo_texloss as lvhippo_texloss
from torch.utils.data import DataLoader

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
    parser.add_argument('--epoch', default=5501, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--data_path', type=str, default='../data/LV_left/train_data.pickle', help='data path to optimize')
    parser.add_argument('--tag', default='tex', help='tex or point')
    parser.add_argument('--cf', default=1, type=float, help='lamda value for cf')
    parser.add_argument('--pm', default=1, type=float, help='lambda value for pm')
    parser.add_argument('--edge', default=1, type=float, help='lambda value for edge')
    parser.add_argument('--norm_con', default=1, type=float, help='lambda value for norm_con')
    parser.add_argument('--lap', default=1, type=float, help='lambda value for lap')
    parser.add_argument('--norm', default=1, type=float, help='lambda value for norm')
    parser.add_argument('--l2', default=1, type=float, help='lambda value for l2')

    return parser.parse_args()
def create_directory(directory_path, log=False):
    try:
        # Create target Directory
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")
        if log:
            raise IOError(f"The file '{directory_path}' exists.")
        else:
            pass
def train_data(data_file):
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    def to_torch(data, key, dtype = np.float32):
        return torch.from_numpy(np.asarray(data[key]).astype(dtype)[np.newaxis, :, :]).cuda()
    
    # Convert vertices and faces to tensors and move to GPU
    vertices = to_torch(data,'vertices', np.float32)
    lv_faces = to_torch(data,'lv_faces', np.int32)
    hippo_faces = to_torch(data,'hippo_faces', np.int32)
    tri1_faces = to_torch(data,'tri1_faces', np.int32)
    tri2_faces = to_torch(data,'tri2_faces', np.int32)
    tri3_faces = to_torch(data,'tri3_faces', np.int32)
    tri4_faces = to_torch(data,'tri4_faces', np.int32)
    tri5_faces = to_torch(data,'tri5_faces', np.int32)
    
    target_lv = to_torch(data,'target_lv', np.float32)
    target1 = to_torch(data,'target1', np.float32) # thalamus
    target2 = to_torch(data,'target2', np.float32) # caudate
    target3 = to_torch(data,'target3', np.float32) # opposite lv
    target4 = to_torch(data,'target4', np.float32) # hippocampus
    target5 = to_torch(data,'target5', np.float32) # the other
    tri = [tri1_faces, tri2_faces, tri3_faces, tri4_faces, tri5_faces]
    target = [target1, target2, target3, target4, target5]
    return vertices, lv_faces, hippo_faces, tri, target, target_lv

def main(args):
    writer = SummaryWriter(log_dir=f"./log/{args.log_dir}")

    def log_string(str):
        logger.info(str)
        print(str)
 
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    create_directory(rf'C:\Users\v1wpark\Documents\Projects\lv-parametric-modelling\voxel2mesh\out\{args.tag}')

    create_directory(rf'C:\Users\v1wpark\Documents\Projects\lv-parametric-modelling\voxel2mesh\log\{args.tag}')

    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    writer = SummaryWriter(log_dir=rf'C:\Users\v1wpark\Documents\Projects\lv-parametric-modelling\voxel2mesh\log\{args.tag}')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    lda = {}
    lda['cf'], lda['pm'], lda['edge'], lda['norm_con'], lda['lap'], lda['norm'], lda['l2'] = args.cf, args.pm, args.edge, args.norm_con, args.lap, args.norm, args.l2
    '''MODEL LOADING'''
    model = PointNetOpt(num_classes=3, input_transform=False, feature_transform=False).to("cuda")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch//4, gamma=0.5)
    global_epoch = 0
    global_step = 0

    '''TRANING'''
    logger.info('Start training...')
    #torch.autograd.set_detect_anomaly(True)
    model.train()
    loss_fn = nn.MSELoss()

    # data preparation
    vertices, lv_faces, hippo_faces, tri, target, target_lv = train_data(args.data_path)
    # Convert vertices and faces to tensors and move to GPU
    orig_vertices = vertices.clone()

    lv_pred_mesh = Meshes(verts=list(vertices), faces=list(lv_faces))
    hippo_pred_mesh = Meshes(verts=list(vertices), faces=list(hippo_faces))

    orig_lv_norm = lv_pred_mesh.verts_normals_packed().clone()
    orig_hippo_norm = hippo_pred_mesh.verts_normals_packed().clone()
    pred_mesh = Meshes(verts=list(vertices), faces=list(lv_faces))

    # Optimization loop
    print(f"{vertices.shape=}")
    for epoch in range(args.epoch):

        optimizer.zero_grad()

        vertices = pred_mesh.verts_list()[0].unsqueeze(0)

        # Forward pass
        pred = model(vertices)

        # Avoid inplace operations by creating new tensors
        verts = vertices.clone() + pred.transpose(2, 1).clone()
        l2_loss = loss_fn(orig_vertices, verts)
        loss, log = lvhippo_texloss(verts, lv_faces, hippo_faces, tri, target, target_lv, epoch)
        loss += l2_loss * 1.5
        # Backward pass
        lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_faces))
        hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
        lv_norm = lv_pred_mesh.verts_normals_packed()
        hippo_norm = hippo_pred_mesh.verts_normals_packed()
        # print(f"{torch.sum(lv_pred_mesh.verts_normals_packed()!=0.0)=}")

        l2_lv_norm_loss = loss_fn(orig_lv_norm, lv_norm)
        l2_hippo_norm_loss = loss_fn(orig_hippo_norm, hippo_norm)
        loss+=(l2_lv_norm_loss+l2_hippo_norm_loss)* 15

        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1
        
        log['l2 loss'] =l2_loss
        log['lv norm loss'] =l2_lv_norm_loss
        log['hippo norm loss'] = l2_hippo_norm_loss
        if epoch % 30 == 0 :
            print(epoch, log)
        
        for key in log.keys():
            writer.add_scalar(key, log[key], global_step=epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch)
        if epoch % 500 == 0 :
            verts_np = verts.detach().cpu().numpy()
            np.save(rf'C:\Users\v1wpark\Documents\Projects\lv-parametric-modelling\voxel2mesh\out\{args.tag}\{args.tag}_{epoch}_loss{loss.detach().cpu().item()}.npy', verts_np)
            
    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)