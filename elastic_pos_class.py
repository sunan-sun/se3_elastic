import numpy as np
from collections import OrderedDict
# from .src.generate_transfer import start_adapting, get_joints

from .src.gaussian_kinematics import GaussianKinematics3D
from .src.find_joints import _get_joints
from .src.IK import solveIK, solveTraj



class elastic_pos_class:
    def __init__(self, Prior_list, Mu_arr, Sigma_arr, x_0, x_att) -> None:

        self.old_gmm = {
            "Prior": np.array(Prior_list),
            "Mu": Mu_arr,
            "Sigma": Sigma_arr
        }

        self.first_joint = x_0
        self.last_joint  = x_att
        



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
        
        new_gmm = {
            "Prior": pi,
            "Mu": mean_arr,
            "Sigma": cov_arr
        }
        # print(anchor_arr)
        # print("new_cov", new_gmm["Sigma"])

        # Generate new traj
        traj_arr, traj_dot_arr = solveTraj(np.copy(new_anchor), M, dt)  # solve for new trajectory
        # pos_and_vel = np.hstack((traj_arr[1:], traj_dot_arr))


        new_pos = traj_arr[1:]
        new_vel = traj_dot_arr
        new_att = new_pos[-1, :]

        return new_pos, new_vel, old_gmm, new_gmm, old_anchor, new_anchor, new_att