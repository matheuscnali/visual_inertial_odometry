import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from models.raft.utils.utils import InputPadder


def activation_function(x):

    fg_probability = 2 / (1 + np.exp(-2 * (x))) - 1
    return fg_probability

def get_motion_vector(optical_flow_model, img_1, img_2, config):

    img_1 = np.array(img_1).astype(np.uint8); img_1 = torch.from_numpy(img_1).permute(2, 0, 1).float()
    img_2 = np.array(img_2).astype(np.uint8); img_2 = torch.from_numpy(img_2).permute(2, 0, 1).float()

    images = torch.stack([img_1, img_2], dim=0)
    images = images.to('cuda')

    padder = InputPadder(images.shape)
    images = padder.pad(images)[0]
    img_1 = images[0, None]
    img_2 = images[1, None]

    flow_low, flow_up = optical_flow_model(img_1, img_2, iters=20, test_mode=True)

    flow_up = padder.unpad(flow_up)

    flow_up = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()
    v_x = flow_up[:, :, 0].flatten()
    v_y = flow_up[:, :, 1].flatten()

    rad = np.sqrt(np.square(v_x) + np.square(v_y))
    rad_max = np.max(rad)
    epsilon = 1e-5
    v_x = (v_x / (rad_max + epsilon)).flatten()
    v_y = (v_y / (rad_max + epsilon)).flatten()

    v_mag = np.sqrt(np.add(np.power(v_x, 2), np.power(v_y, 2)))
    v_ang = np.add(np.degrees(np.arctan(np.divide(v_y, v_x))), 360) % 360
    return np.array([v_x, v_y, v_mag, v_ang])

def get_pca_projection(data):

    data = StandardScaler().fit_transform(data.T)

    pca = PCA(n_components=1)
    pca.fit(data)
    transformed_data = pca.transform(data)

    return transformed_data.flatten()

def motion_from_optical_flow(motion_vector, dynamic_model, config):

    motion_vector_, dynamic_model_ = motion_vector, dynamic_model
    epsilon, prev_residual = config['epsilon'] + 1e-6, 0
    x, y = config['x'], config['y']

    while epsilon > config['epsilon']:
        # Fit a dynamic model to actual motion
        coefficients, r, rank, s = np.linalg.lstsq(a=dynamic_model_, b=motion_vector_, rcond=None)
        epsilon = abs(r - prev_residual)
        prev_residual = r

        # Compute estimated motion, pixel wise motion error and foreground probability
        a, b, c, d, e, f = coefficients
        estimated_motion = np.array(a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f).T
        pixel_wise_motion_error = np.abs(np.subtract(motion_vector, estimated_motion))
        fg_probability = activation_function(pixel_wise_motion_error)

        # Select inlier pixels
        inlier_mask = fg_probability < config['t_motion']
        inliers_index = np.where(inlier_mask == True)

        # Update models
        dynamic_model_ = dynamic_model[inliers_index]
        motion_vector_ = motion_vector[inliers_index]

    return fg_probability.reshape(config['height'], config['width'])

def motion_probability(img_1, img_2, optical_flow_model, dynamic_model, config):

    # Get motion vector
    if img_1.ndim == 2 or img_2.ndim == 2:
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_GRAY2RGB)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_GRAY2RGB)

    motion_vector = get_motion_vector(optical_flow_model, img_1, img_2, config)
    motion_vector = get_pca_projection(motion_vector)

    # Run motion module
    pm = motion_from_optical_flow(motion_vector, dynamic_model, config)
    bg_mask = pm < config['t_motion']

    return bg_mask
