import numpy as np
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation as R
# from .src.generate_transfer import start_adapting
from .src.util import quat_tools

from .src.gaussian_kinematics import GaussianKinematics3D
from .src.find_joints import _get_joints
from .src.IK import solveIK, solveTraj



class elastic_ori_class:
    def __init__(self, Prior_list, Mu_arr, Sigma_arr, first_joint, last_joint) -> None:
        
        K = int(len(Prior_list) / 2) # only update the half of the double cover
        N = 3 #fixed

        att_vec = last_joint
        att_basis = null_space(att_vec.reshape(1, -1))
        # print("normal_vec", normal_vec)
        # print("normal", normal_basis)
        
        normal_basis = []
        
        Prior = np.zeros((K))
        Mu = np.zeros((K, N))
        Sigma = np.zeros((K, N, N))
        for k in range(K):
            normal_basis.append(null_space(Mu_arr[k, :].reshape(1, -1)))


            Prior[k] = Prior_list[k] * 2
            Mu_red = quat_tools.riem_log(att_vec, Mu_arr[k, :])[0, :] # 2d array to 1d
            Mu_k, _, _, _  = np.linalg.lstsq(att_basis, Mu_red, rcond=None)
            Mu[k, :] = Mu_k
            Sigma[k, :, :] = normal_basis[k].T @ Sigma_arr[k, :, :] @ normal_basis[k]
            
            # eigvalue, eigvector = np.linalg.eig(Sigma_arr[k])
            # eigenvalues, eigenvectors = np.linalg.eig(Sigma_arr[k])
            # index_of_min_eigenvalue = np.argmin(np.abs(eigenvalues))
            # normal_vec = eigenvectors[:, index_of_min_eigenvalue]
            # normal_basis = null_space(normal_vec.reshape(1, -1)) 



            print("\nNormal Vector to the Plane:")
            # print(normal_vec)
            print(Mu_arr[k, :])
        # print("old", Sigma_arr)

        self.old_gmm_struct = {
            "Prior": Prior,
            "Mu": Mu.T,
            "Sigma": Sigma
        } #3d

        self.Sigma_arr = Sigma_arr
        self.att_vec = att_vec
        self.att_basis = att_basis

        self.normal_vec = np.copy(last_joint)
        self.normal_basis = normal_basis
        self.first_joint = self.map(first_joint) #3d
        self.last_joint  = self.map(last_joint)  #3d


    def map(self, q):
        """From QUATERNION to 3D"""
        if isinstance(q, R):
            q1 = quat_tools.riem_log(self.att_vec, q.as_quat())[0, :] #2d array to 1d
        elif isinstance(q, np.ndarray):
            q1 = quat_tools.riem_log(self.att_vec, q)[0, :] #2d array to 1d
        elif isinstance(q, list):
            # qq = quat_tools.list_to_arr(q)
            # print(qq.shape)
            # import matplotlib.pyplot as plt
            # M, N = qq.shape
            # fig, axs = plt.subplots(4, 1, figsize=(12, 8))
            # colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime']
            # for k in range(4):
            #     axs[k].scatter(np.arange(M), qq[:, k], s=5, color=colors[k])
            # plt.show()

            q1 = quat_tools.riem_log(R.from_quat(self.att_vec), q).T

        v1, _, _, _ = np.linalg.lstsq(self.att_basis, q1, rcond=None)

        return v1
    

    def inv_map(self, traj):
        """from 3d traj [M, N] to quat: M is number of points"""
        M = traj.shape[0]
        new_ori = [R.identity()] * M
        for i in range(M):
            ori_i_red = traj[i, :]
            ori_i = self.normal_basis @ ori_i_red + self.normal_vec
            ori_i_quat = quat_tools.riem_exp(self.att, ori_i.reshape(1, -1))
            new_ori[i] = R.from_quat(ori_i_quat[0])

        return new_ori


    def _geo_constr(self, x_start, x_end, v_start, v_end):
    
        v1 = self.map(x_start)
        v_start = self.map(v_start[0]) - self.map(v_start[1])
        v1_normed = v_start / np.linalg.norm(v_start)
        v2 = np.cross(v1_normed, np.array([0, 1, 0]))
        v3 = np.cross(v1_normed, v2)
        R_s = np.column_stack((v1_normed, v2, v3))


        u1 = self.map(x_end)
        v_end = self.map(v_end[0]) - self.map(v_end[1])
        u1_normed = v_end / np.linalg.norm(v_end)
        u2 = np.cross(u1_normed, np.array([0, 1, 0]))
        u3 = np.cross(u1_normed, u2)
        R_e = np.column_stack((u1_normed, u2, u3))

        T_s = np.zeros((4, 4))
        T_e = np.zeros((4, 4))

        T_s[:3, :3] = R_s
        T_s[:3, -1] = v1
        T_s[-1, -1] = 1

        T_e[:3, :3] = R_e
        T_e[:3, -1] = u1
        T_e[-1, -1] = 1

        self.T_s = T_s
        self.T_e = T_e



    def start_adapting(self):
        
        # Read variable
        first_joint = self.first_joint
        last_joint  = self.last_joint
        old_gmm_struct = self.old_gmm_struct

        pi     = old_gmm_struct["Prior"]
        mu     = old_gmm_struct["Mu"].T  
        sigma  = old_gmm_struct["Sigma"]

        anchor_arr = _get_joints(mu, sigma, first_joint, last_joint) # anchor from the beginning to the end

        # Update Gaussians
        gk = GaussianKinematics3D(pi, mu, sigma, anchor_arr)
        traj_dis = np.linalg.norm(last_joint - first_joint)
        new_anchor_point = solveIK(anchor_arr, self.T_s, self.T_e, traj_dis, scale_ratio=None)  # solve for new anchor points
        _, mean_arr, cov_arr = gk.update_gaussian_transforms(new_anchor_point)
        

        # Generate new traj
        traj_data_ori, _ = solveTraj(np.copy(new_anchor_point), dt=0.02)  # solve for new trajectory

        M = traj_data_ori.shape[0]
        new_ori = [R.identity()] * M
        for i in range(M):
            ori_i_red = traj_data_ori[i, :]
            ori_i = self.att_basis @ ori_i_red
            ori_i_quat = quat_tools.riem_exp(self.att_vec, ori_i.reshape(1, -1))
            new_ori[i] = R.from_quat(ori_i_quat[0])

        new_ori_out = [R.identity()] * M
        for i in range(M-1):
            new_ori_out[i] = new_ori[i+1]
        new_ori_out[-1] = new_ori[-1]

        K_ori= pi.shape[0]
        Mu = [R.identity()] * K_ori
        for k in range(K_ori):
            mu_k = mean_arr[k, :]
            mu_k = self.att_basis @ mu_k
            # mu_k = self.normal_basis @ mu_k + self.normal_vec
            mu_k = quat_tools.riem_exp(self.att_vec, mu_k.reshape(1, -1))
            Mu[k] = R.from_quat(mu_k[0])

        Sigma = np.zeros((K_ori, 4, 4))
        for k in range(K_ori):
            Sigma[k, :, :] = self.normal_basis[k] @ cov_arr[k, :, :] @ self.normal_basis[k].T
            # Sigma_gt = self.normal_basis[k] @ self.old_gmm_struct["Sigma"][k] @ self.normal_basis.T
            print("new", Sigma[k])
            # print("new", Sigma[k])

            # print("old", Sigma_gt)
            print("gt", self.Sigma_arr[k])

        new_gmm = {
            "Mu": Mu,
            "Sigma": Sigma,
            "Prior": pi
        }

        return new_ori, new_ori_out, old_gmm_struct, new_gmm, anchor_arr, new_anchor_point, new_ori[-1]

