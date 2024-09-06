import numpy as np
from collections import OrderedDict
# from .src.generate_transfer import start_adapting, get_joints

from .src.gaussian_kinematics import GaussianKinematics3D
from .src.find_joints import _get_joints
from .src.IK import solveIK, solveTraj



class elastic_pos_class:
    def __init__(self, Prior_list, Mu_arr, Sigma_arr, first_joint, last_joint) -> None:
        K = len(Prior_list)
        N = 3

        self.old_gmm_struct = {
            "Prior": np.array(Prior_list),
            "Mu": Mu_arr.T,
            "Sigma": Sigma_arr
        }

        self.first_joint = first_joint
        self.last_joint  = last_joint
        



    def _geo_constr(self, x_start, x_end, v_start, v_end):
        v1 = v_start
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(v1, np.array([0, 1, 0]))
        v3 = np.cross(v1, v2)
        R_s = np.column_stack((v1, v2, v3))

        u1 = v_end
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(u1, np.array([0, 1, 0]))
        u3 = np.cross(u1, u2)
        R_e = np.column_stack((u1, u2, u3))

        T_s = np.zeros((4, 4))
        T_e = np.zeros((4, 4))

        T_s[:3, :3] = R_s
        T_s[:3, -1] = x_start
        T_s[-1, -1] = 1

        T_e[:3, :3] = R_e
        T_e[:3, -1] = x_end
        T_e[-1, -1] = 1

        self.T_s = T_s
        self.T_e = T_e



    # def get_joints(self):
    #     return get_joints([self.data.T], self.old_gmm_struct)        

        

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
        new_gmm = {
            "Mu": mean_arr.T,
            "Sigma": cov_arr,
            "Prior": pi
        }
        # print(anchor_arr)
        print("new_cov", new_gmm["Sigma"])

        # Generate new traj
        plot_traj, traj_dot_arr = solveTraj(np.copy(new_anchor_point), dt=0.02)  # solve for new trajectory
        pos_and_vel = np.vstack((plot_traj[1:].T, traj_dot_arr.T))

        return [pos_and_vel], old_gmm_struct, new_gmm, new_anchor_point, anchor_arr, plot_traj[-1, :]