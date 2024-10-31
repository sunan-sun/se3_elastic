import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.transform import Rotation as R
import random


font = {'family' : 'Times New Roman',
         'size'   : 18
         }
mpl.rc('font', **font)




def demo_vs_adjust(demo, adjust, old_anchor, new_anchor, q_in, new_ori):

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    ax.plot(demo[:, 0], demo[:, 1], demo[:, 2], 'o', color='gray', alpha=0.2, markersize=1.5, label="Demonstration")
    ax.plot(adjust[:, 0], adjust[:, 1], adjust[:, 2], 'o', color='red', alpha=0.2, markersize=1.5, label="Adjusted")

    ax.scatter(demo[0, 0], demo[0, 1], demo[0, 2], 'o', facecolors='none', edgecolors='magenta',linewidth=2,  s=100, label="Initial")
    ax.scatter(demo[-1, 0], demo[-1, 1], demo[-1, 2], marker=(8, 2, 0), color='k',  s=100, label="Target")

    for i in range(old_anchor.shape[0]):
        ax.scatter(old_anchor[i, 0], old_anchor[i, 1], old_anchor[i, 2], 'o', facecolors='none', edgecolors='black',linewidth=2,  s=100)

    for i in range(new_anchor.shape[0]):
        ax.scatter(new_anchor[i, 0], new_anchor[i, 1], new_anchor[i, 2], 'o', facecolors='none', edgecolors='red',linewidth=2,  s=100)

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min) * 0.4
    for i in np.linspace(0, len(q_in), num=10, endpoint=False, dtype=int):
        r = q_in[i]
        loc = demo[i, :]
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                            colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, alpha=0.6,linewidth=1)

    if len(new_ori)!=0:
        for i in np.linspace(0, len(new_ori), num=10, endpoint=False, dtype=int):
            r = new_ori[i]
            loc = adjust[i, :]
            # loc = demo[i, :]

            for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                                colors)):
                line = np.zeros((2, 3))
                line[1, j] = scale
                line_rot = r.apply(line)
                line_plot = line_rot + loc
                ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c,  linewidth=1)

    ax.axis('equal')
    ax.legend(ncol=4, loc="upper center")

    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))





def demo_vs_adjust_gmm(p_in, demo, gmm, old_gmm_struct, gmm_struct, new_ori=[]):

    label = gmm.assignment_arr
    K     = gmm.K

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o', color=color_mapping[:], s=1, alpha=0.4, label="Demonstration")

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB

    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min)/4
    for k in range(K):
        loc = old_gmm_struct["Mu"][:, k]
        ax.text(loc[0], loc[1], loc[2], str(k + 1), fontsize=20)

    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o', color=color_mapping[:], s=1, alpha=0.4, label="Demonstration")

    for k in range(K):
        loc = gmm_struct["Mu"][:, k]
        ax.text(loc[0], loc[1], loc[2], str(k + 1), fontsize=20)

    ax.axis('equal')

    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))



def plot_gmm(p_in, gmm, gmm_struct):

    label = gmm.assignment_arr
    K     = gmm.K

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o', color=color_mapping[:], s=1, alpha=0.4, label="Demonstration")

    for k in range(K):
        loc = gmm_struct["Mu"][:, k]
        ax.text(loc[0], loc[1], loc[2], str(k + 1), fontsize=20)

    ax.axis('equal')

    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))




def ori_debug(ori_tra, new_ori_tra, old_anchor, new_anchor, gmm, new_gmm):

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')


    ax.plot(ori_tra[:, 0], ori_tra[:, 1], ori_tra[:, 2], 'o', color='gray', alpha=0.2, markersize=1.5, label="Demonstration")
    ax.plot(new_ori_tra[:, 0], new_ori_tra[:, 1], new_ori_tra[:, 2], 'o', color='red', alpha=0.2, markersize=1.5, label="Adjusted")

    for i in range(old_anchor.shape[0]):
        ax.scatter(old_anchor[i, 0], old_anchor[i, 1], old_anchor[i, 2], 'o', facecolors='none', edgecolors='black',linewidth=2,  s=100)
        # ax.text(old_anchor[i, 0], old_anchor[i, 1], old_anchor[i, 2], str(i + 1), fontsize=20)

    for i in range(new_anchor.shape[0]):
        ax.scatter(new_anchor[i, 0], new_anchor[i, 1], new_anchor[i, 2], 'o', facecolors='none', edgecolors='red',linewidth=2,  s=100)

    ax.axis('equal')

    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))