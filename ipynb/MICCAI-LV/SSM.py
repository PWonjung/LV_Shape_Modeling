import os
import torch
import open3d as o3d

from tqdm import tqdm
import numpy as np
import trimesh
import pyvista as pv
import numbers
from pytorch3d.loss import chamfer_distance
import matplotlib.pyplot as plt
import pickle
# 일단은 파일 하나에 다 때려박아놓음

class SSMPCA:
    def __init__(self, correspondences):
        """
        Compute the SSM based on eigendecomposition.
        Args:
            correspondences:    Corresponded shapes as a torch.Tensor
        """
        self.device = correspondences.device
        self.mean = torch.mean(correspondences, dim=0)

        data_centered = correspondences - self.mean
        cov_dual = torch.matmul(data_centered, data_centered.T) / (
            data_centered.shape[0] - 1
        )

        evals, evecs = torch.linalg.eigh(cov_dual)
        evecs = torch.matmul(data_centered.t(), evecs)
        # Normalize the col-vectors
        evecs /= torch.sqrt(torch.sum(evecs ** 2, dim=0))

        # Sort
        idx = torch.argsort(evals, descending=True)
        evecs = evecs[:, idx]
        evals = evals[idx]

        # Remove the last eigenpair (it should have zero eigenvalue)
        self.variances = evals[:-1]
        self.modes_norm = evecs[:, :-1]
        # Compute the modes scaled by corresp. std. dev.
        self.modes_scaled = self.modes_norm * torch.sqrt(self.variances)
        self.modes_scaled = self.modes_scaled.to(torch.float32)
        self.length = evecs.shape[0]

    def generate_random_samples(self, n_samples=1, n_modes=None):
        """
        Generate random samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples as torch.Tensor
        """
        if n_modes is None:
            n_modes = self.modes_scaled.shape[1]
        weights = torch.randn(n_samples, n_modes).to(self.device)
        samples = self.mean + torch.matmul(weights, self.modes_scaled.t()[:n_modes])
        return samples.squeeze()

    def get_reconstruction(self, shape, n_modes=None):
        """
        Project shape into the SSM to get a reconstruction
        Args:
            shape:      shape to reconstruct as torch.Tensor
            n_modes:    number of modes to use. If None, all relevant modes are used
        Returns:
            data_proj:  projected data as reconstruction as torch.Tensor
        """
        shape = shape.view(-1)
        data_proj = shape - self.mean
        if n_modes:
            # restrict to max number of modes
            if n_modes > self.length:
                n_modes = self.modes_scaled.shape[1]
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        evecs_t = evecs.t()
        data_proj_re = data_proj.view(-1, 1)
        weights = torch.matmul(evecs_t, data_proj_re)
        data_proj = self.mean + torch.matmul(weights.t(), evecs_t)
        data_proj = data_proj.view(-1, 3)
        return data_proj.float()


def build_ssm(dataset, method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_path = f"../data/{dataset}"
    # recon data path에는 template을 변형하여 만든 mesh들이 들어있음
    recon_data_path = f"data/{dataset}/{method}"
    
    # 원래는 여기서 template을 골라야 하지만, 여기서는 deformed template을 선계산하여서 생략됨
    
    # 주의 : vertex 개수 하드코딩됨
    # deforming template to all training shapes to run PCA
    deformed_training_shapes = torch.empty(0, 2000, 3).to(device)
    
    for data in os.listdir(os.path.join(recon_data_path, "train", "recon")):
        # deformed_verts, name = self.deform_template(data)
        deform_verts = o3d.io.read_triangle_mesh(os.path.join(recon_data_path, "train", "recon", data)).vertices
        deform_verts = torch.tensor(deform_verts).to(device)
        deformed_training_shapes = torch.cat((deformed_training_shapes, deform_verts.unsqueeze(0)), dim=0)
    deformed_training_shapes = deformed_training_shapes.reshape(deformed_training_shapes.shape[0], -1)
        
    # build SSM using PCA
    ssm_model = SSMPCA(deformed_training_shapes)
    
    # deform test shapes (for calculating generalization)
    deformed_testing_shapes = torch.empty(0, 2000, 3).to(device)
    # generalization calculation에서 face 정보가 필요함
    # 이 deformed mesh들은 face 정보가 template과 일치하게 한 후 아무 deformed mesh의 face 정보를 가져옴
    template = None
    for data in os.listdir(os.path.join(recon_data_path, "test", "recon")):
        # deformed_verts, name = self.deform_template(data)
        deform_verts = o3d.io.read_triangle_mesh(os.path.join(recon_data_path, "test", "recon", data)).vertices
        deform_verts = torch.tensor(deform_verts).to(device)
        deformed_testing_shapes = torch.cat((deformed_testing_shapes, deform_verts.unsqueeze(0)), dim=0)
        template = o3d.io.read_triangle_mesh(os.path.join(recon_data_path, "test", "recon", data))
    deformed_testing_shapes = deformed_testing_shapes.reshape(deformed_testing_shapes.shape[0], -1)
    
    output_path = os.path.join("output", dataset, method)
    os.makedirs(output_path, exist_ok=True)
    
    calculate_generalization(ssm_model, os.path.join(dataset_path, "test_meshes"), deformed_testing_shapes, None, device, output_path, template)
    calculate_specificity(ssm_model, os.path.join(dataset_path, "train_meshes"), None, device, output_path)
    calculate_compactness(ssm_model, os.path.join(dataset_path, "train_meshes"), None, device, output_path)
    

def calculate_generalization(ssm_model, test_data_path, deformed_testing_shapes, logger, device, output_path, template):
    surface_distance = SurfaceDistance()

    generalizations_mean = []
    generalizations_std = []
    print(f'Calculating Generalization')

    for mode in tqdm(range(1, ssm_model.variances.shape[0] + 1)):
        generalizations_per_mode = []

        for index, test_data in enumerate(os.listdir(test_data_path)):
            # original_verts = (test_data['verts'].to(device) * test_data['face_area'].to(device)).float()
            original_verts = np.asarray(read_vtk_mesh(os.path.join(test_data_path, test_data)).vertices)
            original_faces = np.asarray(read_vtk_mesh(os.path.join(test_data_path, test_data)).triangles)
            recon_deformed_shape = (ssm_model.get_reconstruction(deformed_testing_shapes[index], n_modes=mode)
                                    .reshape(1, -1, 3).to(device)).float()

            original_mesh = trimesh.Trimesh(vertices=to_numpy(original_verts), faces=to_numpy(original_faces))
            template_face = np.asarray(template.triangles)
            recon_mesh = trimesh.Trimesh(vertices=to_numpy(recon_deformed_shape),
                                         faces=to_numpy(template_face))
            surf_dist = surface_distance(original_mesh, recon_mesh)[0]
            generalizations_per_mode.append(surf_dist)


        generalization_per_mode_mean = np.mean(generalizations_per_mode)
        generalization_per_mode_std = np.std(generalizations_per_mode)
        generalizations_mean.append(generalization_per_mode_mean)
        generalizations_std.append(generalization_per_mode_std)
        print(
            f'Generalizations for mode {mode} is {generalization_per_mode_mean:.4f} +/- {generalization_per_mode_std:.4f}')

    result_path = os.path.join(output_path, "generality.png")
    generalizations_mean = np.array(generalizations_mean)
    generalizations_std = np.array(generalizations_std)
    plot_with_std(np.array(list(range(1, ssm_model.variances.shape[0] + 1))),
                  generalizations_mean, generalizations_std,
                  "Generality in mm", result_path)
    np.save(os.path.join(output_path, "generalizations_mean.npy"), generalizations_mean)
    np.savetxt(os.path.join(output_path, "generalizations_mean.txt"), generalizations_mean)
    np.save(os.path.join(output_path, "generalizations_std.npy"), generalizations_std)
    np.savetxt(os.path.join(output_path, "generalizations_std.txt"), generalizations_std)




class SurfaceDistance():
    """This class calculates the symmetric vertex to surface distance of two
    trimesh meshes.
    """

    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Args:
          A: trimesh mesh
          B: trimesh mesh
        """
        _, A_B_dist, _ = trimesh.proximity.closest_point(A, B.vertices)
        _, B_A_dist, _ = trimesh.proximity.closest_point(B, A.vertices)
        distance = .5 * np.array(A_B_dist).mean() + .5 * \
            np.array(B_A_dist).mean()

        return np.array([distance])

def calculate_generalization_point(ssm_model, test_data_path, deformed_testing_shapes, logger, device, output_path):
    
    generalization_mean = []
    generalization_std = []
    print(f'Calculating Generalization')
    
    for mode in tqdm(range(1, ssm_model.variances.shape[0] + 1)):
        generalization_per_mode = []
        
        for index, test_data in enumerate(os.listdir(test_data_path)):
            if index<10:
                original_verts = np.loadtxt(os.path.join(test_data_path, test_data))
                # original_faces = np.asarray(read_vtk_mesh(os.path.join(test_data_path, test_data)).triangles)
                recon_deformed_shape = (ssm_model.get_reconstruction(deformed_testing_shapes[index], n_modes=mode)
                                        .reshape(1, -1, 3).to(device)).float()
                
                original_verts = torch.tensor(original_verts).reshape(1, -1, 3).to(device).float()
                recon_deformed_shape = recon_deformed_shape.reshape(1, -1, 3).to(device).float()
                
                dist, _ = chamfer_distance(original_verts, recon_deformed_shape, point_reduction=None, batch_reduction=None)
                surf_dist = 0.5 * (dist[0].sqrt().mean(dim=1) + dist[1].sqrt().mean(dim=1))
                surf_dist = to_numpy(surf_dist)
                
                generalization_per_mode.append(surf_dist)
            
        generalization_per_mode_mean = np.mean(generalization_per_mode)
        generalization_per_mode_std = np.std(generalization_per_mode)
        generalization_mean.append(generalization_per_mode_mean)
        generalization_std.append(generalization_per_mode_std)
        print(f'Generalizations for mode {mode} is {generalization_per_mode_mean:.4f} +/- {generalization_per_mode_std:.4f}')
        
    result_path = os.path.join(output_path, "generality.png")
    generalization_mean = np.array(generalization_mean)
    generalization_std = np.array(generalization_std)
    plot_with_std(np.array(list(range(1, ssm_model.variances.shape[0] + 1))),
                  generalization_mean, generalization_std,
                  "Generality in mm", result_path)
    np.save(os.path.join(output_path, "generalizations_mean.npy"), generalization_mean)
    np.savetxt(os.path.join(output_path, "generalizations_mean.txt"), generalization_mean)
    np.save(os.path.join(output_path, "generalizations_std.npy"), generalization_std)
    np.savetxt(os.path.join(output_path, "generalizations_std.txt"), generalization_std)  
    
def calculate_specificity(ssm_model, train_data_path, logger, device, output_path):
    n_samples = 1000
    specificity_mean = []
    specificity_std = []
    print(f'Calculating Specificity...')

    for mode in tqdm(range(1, ssm_model.variances.shape[0] + 1)):
        samples = ssm_model.generate_random_samples(n_samples=n_samples, n_modes=mode)
        samples = samples.reshape(n_samples, -1, 3).to(device)
        samples = samples - samples.mean(dim=1, keepdim=True)

        distances = np.zeros((n_samples, len(os.listdir(train_data_path))))

        for index, data in enumerate(os.listdir(train_data_path)):

            tgt_pt = np.loadtxt(os.path.join(train_data_path, data))
            target = torch.tensor(tgt_pt).to(device)
            ## pick random 3000 points
            # tgt_pt = tgt_pt[np.random.choice(tgt_pt.shape[0], 3000, replace=False)]
            # target = torch.tensor(np.asarray(read_vtk_mesh(os.path.join(train_data_path, data)).vertices)).to(device)
            target = torch.tensor(tgt_pt).to(device)
            print(target)
            # print(target.shape)
            target = target.repeat(n_samples, 1, 1)
            
            loss, _ = chamfer_distance(target.float(), samples.float(), point_reduction=None, batch_reduction=None)
            distance = 0.5 * (loss[0].sqrt().mean(dim=1) + loss[1].sqrt().mean(dim=1))

            distance = to_numpy(distance)
            distances[:, index] = distance

        specificity_mean_value = distances.min(1).mean()
        specificity_std_value = distances.min(1).std()
        specificity_mean.append(specificity_mean_value)
        specificity_std.append(specificity_std_value)
        print(f'Specificity for mode {mode} is {specificity_mean_value:.10f} +/- {specificity_std_value:.10f}')

    result_path = os.path.join(output_path, "specificity.png")
    specificity_mean = np.array(specificity_mean)
    specificity_std = np.array(specificity_std)
    plot_with_std(np.array(list(range(1, ssm_model.variances.shape[0] + 1))),
                  specificity_mean, specificity_std,
                  "Specificity in mm", result_path)
    # np.save(os.path.join(output_path, "specificity_mean.npy"), specificity_mean)
    # np.save(os.path.join(output_path, "specificity_std.npy"), specificity_std)
    return samples
    
def calculate_compactness(ssm_model, train_data_path, logger, device, output_path):
    n_samples = 1000
    compactness_full = []
    print(f'Calculating Compactness...')
    
    eigenvalues = ssm_model.variances
    sum_eigenvalues = torch.sum(eigenvalues)
    
    temp = 0
    for mode in tqdm(range(1, ssm_model.variances.shape[0] + 1)):
        temp += eigenvalues[mode - 1]
        compactness = temp / sum_eigenvalues
        compactness_full.append(to_numpy(compactness))
        print(f'Compactness for mode {mode} is {compactness:.10f}')

    result_path = os.path.join(output_path, "compactness.png")
    compactness_full = np.array(compactness_full)
    plot_with_std(np.array(list(range(1, ssm_model.variances.shape[0] + 1))),
                  compactness_full, 0,
                  "Compactness", result_path, y_label='Explained variance')
    # np.save(os.path.join(output_path, "compactness.npy"), compactness_full)
    print(compactness_full)
    
# vtk 파일 읽어서 o3d mesh로 변환    
def read_vtk_mesh(file_path):
    mesh = pv.read(file_path)
    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def to_numpy(tensor, squeeze=True):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        if squeeze:
            tensor = tensor.squeeze()
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, numbers.Number):
        return np.array([tensor])
    else:
        raise NotImplementedError()


def plot_with_std(x, y_mean, y_std, title, path, y_label='Error'):
    # Plot the mean line
    plt.plot(x, y_mean, label='Mean')

    # Plot the shaded region for standard deviation
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label='std')

    # Add labels and legend
    plt.xlabel('Modes')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    # save plot
    plt.savefig(path)
    plt.close()
