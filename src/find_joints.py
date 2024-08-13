import numpy as np


def _get_product_gaussian(mu1, mu2, sigma1, sigma2):
    combine_sigma = np.linalg.inv(np.linalg.inv(sigma1) + np.linalg.inv(sigma2))
    max_pos = combine_sigma @ (np.linalg.inv(sigma1) @ mu1 + np.linalg.inv(sigma2) @ mu2)
    max_pos = max_pos.flatten()

    return max_pos, combine_sigma


def _get_joints(mu, sigma, start, end):
    jnt_arr = []

    for i in range(len(mu)-1):
        max_pos, _  = _get_product_gaussian(mu[i], mu[i+1], sigma[i], sigma[i+1])
        jnt_arr.append(max_pos)

    jnt_arr.insert(0, start)
    jnt_arr.append(end)

    return np.array(jnt_arr)