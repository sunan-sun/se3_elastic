import numpy as np
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation as R
from .src.generate_transfer import start_adapting
from .src.util import quat_tools



def _rearrange_clusters(Prior, Mu, Sigma, att):
    """Transpose Mu to be fixed"""

    dist_list = [np.linalg.norm(mu - att) for mu in Mu.T]
    idx = np.array(dist_list).argsort()

    ds_gmm = {
        "Prior": Prior[idx],
        "Mu": Mu[:, idx],
        "Sigma": Sigma[idx, :, :]
    }
    
    return ds_gmm




class elastic_ori_class:
    def __init__(self, Prior_list, Mu_list, Sigma_list, q_att, q_in, q_out) -> None:
        
        self.q_in_ori = q_in

        normal_vec = q_att.as_quat()
        normal_basis = null_space(normal_vec.reshape(1, -1))

        q_in_att = quat_tools.riem_log(q_att, q_in)     # q_in projected onto tangent plane w.r.t. q_att
        q_out_att = quat_tools.riem_log(q_att, q_out)

        q_in_red, _, _, _ = np.linalg.lstsq(normal_basis, q_in_att.T, rcond=None) # projected q_in expressed in R^3 [3, M], where M is number of points
        q_out_red, _, _, _ = np.linalg.lstsq(normal_basis, q_out_att.T, rcond=None) 
        
        self.normal_vec = normal_vec
        self.normal_basis = normal_basis
        
        self.q_att = q_att
        self.q_in  = q_in_red.T
        self.q_out = q_out_red.T
        self.data = np.hstack((self.q_in, self.q_out)) # stacked and trasposed orientation input/output, [M, 6]

        K = int(len(Prior_list) / 2)
        N = 3

        Prior = np.zeros((K))
        Mu = np.zeros((N, K))
        Sigma = np.zeros((K, N, N))

        for k in range(K):
            Prior[k] = Prior_list[k] * 2
            Mu_red = quat_tools.riem_log(q_att, Mu_list[k])[0]
            Mu_k, _, _, _  = np.linalg.lstsq(normal_basis, Mu_red, rcond=None)
            # Mu_k, _, _, _  = np.linalg.lstsq(normal_basis, Mu_list[k].as_quat(), rcond=None)
            Mu[:, k] = Mu_k
            Sigma[k, :, :] = normal_basis.T @ Sigma_list[k] @ normal_basis

        self.old_gmm_struct = _rearrange_clusters(Prior, Mu, Sigma, np.zeros((3, )))


    def _geo_constr(self):

        # q1 = R.identity()
        q1 = self.q_in_ori[0] * R.from_euler('xyz', [2.5, 0.2, 2.1])
        q1 = quat_tools.riem_log(self.q_att, q1)
        v1, _, _, _ = np.linalg.lstsq(self.normal_basis, q1[0], rcond=None) 
        
        v11 = self.q_in[25] - self.q_in[0]
        v1_normed = v11 / np.linalg.norm(v1)
        v2 = np.cross(v1_normed, np.array([0, 1, 0]))
        v3 = np.cross(v1_normed, v2)
        R_s = np.column_stack((v1_normed, v2, v3))

        u1 = self.q_in[-1] - self.q_in[-25]
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(u1, np.array([0, 1, 0]))
        u3 = np.cross(u1, u2)
        R_e = np.column_stack((u1, u2, u3))

        T_s = np.zeros((4, 4))
        T_e = np.zeros((4, 4))

    
        T_s[:3, :3] = R_s
        # T_s[:3, -1] = v1
        T_s[:3, -1] = self.q_in[0]
        T_s[-1, -1] = 1

        T_e[:3, :3] = R_e
        T_e[:3, -1] = v1
        # T_e[:3, -1] = self.q_in[-1]

        T_e[-1, -1] = 1

        self.T_s = T_s
        self.T_e = T_e



    def start_adapting(self):
        self._geo_constr()
        traj_data, gmm_struct, old_anchor, new_anchor = start_adapting([self.data.T], self.old_gmm_struct, self.T_s, self.T_e, scale_ratio=1)

        return traj_data, self.old_gmm_struct ,gmm_struct, old_anchor, new_anchor
