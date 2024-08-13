import numpy as np
from collections import OrderedDict
from .src.generate_transfer import start_adapting, get_joints



def _rearrange_clusters(Prior, Mu, Sigma, att, assignment_arr):
    """Transpose Mu to be fixed....alternative method to order"""

    # dist_list = [-np.linalg.norm(mu - att) for mu in Mu.T] # negative norm hence sort in descending order
    # idx = np.array(dist_list).argsort()

    idx  = list(OrderedDict.fromkeys(assignment_arr))

    ds_gmm = {
        "Prior": Prior[idx],
        "Mu": Mu[:, idx],
        "Sigma": Sigma[idx, :, :]
    }

    return ds_gmm




class elastic_pos_class:
    def __init__(self, Prior_list, Mu_list, Sigma_list, p_att, p_in, p_out, assignment_arr) -> None:
        self.p_in = p_in
        self.p_out = p_out
        self.data = np.hstack((p_in, p_out))

        K = len(Prior_list)
        N = 3

        Prior = np.zeros((K))
        Mu = np.zeros((N, K))
        Sigma = np.zeros((K, N, N))

        for k in range(K):
            Prior[k] = Prior_list[k]
            Mu[:, k] = Mu_list[k, :]
            Sigma[k, :, :] = Sigma_list[k, :, :]

        self.old_gmm_struct = _rearrange_clusters(Prior, Mu, Sigma, p_att, assignment_arr)

        self.p_att = p_att


    def _geo_constr(self):
        # v1 = np.random.rand(3,)
        v1 = self.p_in[25] - self.p_in[0]
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(v1, np.array([0, 1, 0]))
        v3 = np.cross(v1, v2)
        R_s = np.column_stack((v1, v2, v3))

        u1 = self.p_in[-1] - self.p_in[-100]
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(u1, np.array([0, 1, 0]))
        u3 = np.cross(u1, u2)
        R_e = np.column_stack((u1, u2, u3))

        T_s = np.zeros((4, 4))
        T_e = np.zeros((4, 4))

        T_s[:3, :3] = R_s
        T_s[:3, -1] = self.p_in[0]
        T_s[-1, -1] = 1

        T_e[:3, :3] = R_e
        T_e[:3, -1] = self.p_in[-1]
        T_e[-1, -1] = 1

        self.T_s = T_s
        self.T_e = T_e


    def get_joints(self):

        return get_joints([self.data.T], self.old_gmm_struct)        



    def start_adapting(self):
        self._geo_constr()
        traj_data, gmm_struct, old_anchor, new_anchor = start_adapting([self.data.T], self.old_gmm_struct, self.T_s, self.T_e)

        gmm_struct['Mu'][:, -1] = self.p_att # move the last Gaussian mean to attractor
        return traj_data, self.old_gmm_struct ,gmm_struct, old_anchor, new_anchor
