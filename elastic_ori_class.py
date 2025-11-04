import numpy as np
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation as R
# from .src.generate_transfer import start_adapting
from .src.util import quat_tools

from .src.gaussian_kinematics import GaussianKinematics3D
from .src.find_joints import _get_joints
from .src.IK import solveIK, solveTraj



class elastic_ori_class:
    def __init__(self, Prior_list, Mu_arr, Sigma_arr, start_pt, end_pt) -> None:
        """ Convert Mu, Sigma, start_pt, and end_pt from 4D to 3D representation """

        K = int(len(Prior_list) / 2) # only update the half of the double cover
        N = 3 # fixed
        Prior = np.zeros((K))
        Mu = np.zeros((K, N))
        Sigma = np.zeros((K, N, N))

        q_att = R.from_quat(end_pt)

        normal_basis = [null_space(Mu_arr[k, :].reshape(1, -1)) for k in range(K)] # basis for covariance

        # att_basis = null_space(q_att.as_quat().reshape(1, -1))
        # Mu_att = quat_tools.riem_log(q_att, Mu_arr[:K,])  
        # Mu_att_3d = quat_tools.map(q_att, Mu_att) 
        # normal_basis = []

        for k in range(K):
            Prior[k] = Prior_list[k] * 2
            Mu[k, :] = self.map(q_att, Mu_arr[k, :])
            Sigma[k, :, :] = normal_basis[k].T @ Sigma_arr[k, :, :] @ normal_basis[k]

        self.old_gmm = {
            "Prior": Prior,
            "Mu": Mu,
            "Sigma": Sigma
        }

        self.q_att = q_att
        self.Sigma_arr = Sigma_arr

        # can be removed later
        self.att_vec = end_pt
        self.att_basis = null_space(self.att_vec.reshape(1, -1))

        # self.normal_vec = np.copy(last_joint)

        self.normal_basis = normal_basis
        self.first_joint = self.map(q_att, start_pt) 
        self.last_joint  = self.map(q_att, end_pt)  


    def map(self, q_att, q):
        if isinstance(q, R):
            q1 = quat_tools.riem_log(q_att, q)
        elif isinstance(q, np.ndarray):
            q1 = quat_tools.riem_log(q_att, R.from_quat(q))

        v1 = quat_tools.map(q_att, q1)
        return v1
    

    def inv_map(self, q_att, q):
        
        q_tangent = quat_tools.inv_map(q_att, q)
        new_ori = quat_tools.riem_exp(q_att, q_tangent)

        return new_ori


    def _geo_constr(self, x_start, x_end, v_start, v_end):
        q_att = self.q_att

        v1 = self.map(q_att, x_start)
        v_start = self.map(q_att, v_start[0]) - self.map(q_att, v_start[1])
        v1_normed = v_start / np.linalg.norm(v_start)
        v2 = np.cross(v1_normed, np.array([0, 1, 0]))
        v3 = np.cross(v1_normed, v2)
        R_s = np.column_stack((v1_normed, v2, v3))


        u1 = self.map(q_att, x_end)
        v_end = self.map(q_att, v_end[0]) - self.map(q_att, v_end[1])
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



    def start_adapting(self, M, dt):
        
        # Read variable
        first_joint = self.first_joint
        last_joint  = self.last_joint
        old_gmm = self.old_gmm

        pi     = old_gmm["Prior"]
        mu     = old_gmm["Mu"]  
        sigma  = old_gmm["Sigma"]

        old_anchor = _get_joints(mu, sigma, first_joint, last_joint) # anchor from the beginning to the end

        # Update Gaussians
        gk = GaussianKinematics3D(pi, mu, sigma, old_anchor)
        traj_dis = np.linalg.norm(last_joint - first_joint)
        new_anchor = solveIK(old_anchor, self.T_s, self.T_e, traj_dis, scale_ratio=None)  # solve for new anchor points
        _, mean_arr, cov_arr = gk.update_gaussian_transforms(new_anchor)
        

        # Generate new traj
        traj_arr, _ = solveTraj(np.copy(new_anchor), M, dt)  # M, N

        # M = traj_data_ori.shape[0]
        # new_ori = [R.identity()] * M
        # for i in range(M):
        #     ori_i_red = traj_data_ori[i, :]
        #     ori_i = self.att_basis @ ori_i_red
        #     ori_i_quat = quat_tools.riem_exp(self.att_vec, ori_i.reshape(1, -1))
        #     new_ori[i] = R.from_quat(ori_i_quat[0])

        # new_ori_arr = self.inv_map(self.q_att, traj_data_ori)
        # new_ori = [R.from_quat(new_ori_arr[i, :]) for i in range(M)]

        ori_list = [R.from_quat(q) for q in self.inv_map(self.q_att, traj_arr)]

        # new_ori_out = [R.identity()] * M
        # for i in range(M-1):
        #     new_ori_out[i] = new_ori[i+1]
        # new_ori_out[-1] = new_ori[-1]

        Mu = [R.from_quat(q) for q in self.inv_map(self.q_att, mean_arr)]
        Sigma = np.array([self.normal_basis[k] @ cov_arr[k, :, :] @ self.normal_basis[k].T for k in range(pi.shape[0])])

        # K_ori= pi.shape[0]
        # Mu = [R.identity()] * K_ori
        # for k in range(K_ori):
        #     mu_k = mean_arr[k, :]
        #     mu_k = self.att_basis @ mu_k
        #     # mu_k = self.normal_basis @ mu_k + self.normal_vec
        #     mu_k = quat_tools.riem_exp(self.att_vec, mu_k.reshape(1, -1))
        #     Mu[k] = R.from_quat(mu_k[0])

        # Sigma = np.zeros((K_ori, 4, 4))
        # for k in range(K_ori):
        #     Sigma[k, :, :] = self.normal_basis[k] @ cov_arr[k, :, :] @ self.normal_basis[k].T
            # Sigma_gt = self.normal_basis[k] @ self.old_gmm_struct["Sigma"][k] @ self.normal_basis.T
            # print("new", Sigma[k])
            # print("new", Sigma[k])

            # print("old", Sigma_gt)
            # print("gt", self.Sigma_arr[k])

        new_gmm = {
            "Prior": pi,
            "Mu": Mu,
            "Sigma": Sigma
        }

        new_ori     = ori_list[:-1]
        new_ori_out = ori_list[1:]
        new_att = new_ori[-1]
        
        return traj_arr, new_ori, new_ori_out, old_gmm, new_gmm, old_anchor, new_anchor, new_att

