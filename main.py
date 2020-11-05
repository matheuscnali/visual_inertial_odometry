import argparse
import cv2
import yaml
import pykitti
import numpy as np
from sklearn.preprocessing import normalize

from matcher.matching import Matching
from matcher.utils import frame2tensor


class VisualOdometry:

    def __init__(self, config, device):

        self.device = device
        self.matcher = Matching(config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        self.inlier_threshold = config['inlier_threshold']

    def get_matches(self, image):

        # Get matches
        frame_tensor = frame2tensor(image, self.device)
        sg_current = self.matcher({**self.sg_reference, 'image1': frame_tensor})
        matches = sg_current['matches0'][0].cpu().numpy()

        # Select valid key points from reference and current
        valid = matches > -1
        kpts0 = self.sg_reference['keypoints0'][0].cpu().numpy()
        kpts1 = sg_current['keypoints1'][0].cpu().numpy()

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        # Update reference
        self.sg_reference = {k + '0': sg_current[k + '1'] for k in self.keys}
        self.sg_reference['image0'] = frame_tensor

        return mkpts0, mkpts1

    def set_sg_reference(self, image):

        frame_tensor = frame2tensor(image, self.device)
        sg_reference = self.matcher.superpoint({'image': frame_tensor})
        sg_reference = {k + '0': sg_reference[k] for k in self.keys}
        sg_reference['image0'] = frame_tensor
        self.sg_reference = sg_reference

    def filter_matches(self, velocity, matches, k, inv_k):

        matches_flow = np.subtract(matches[1], matches[0])
        matches_flow_normalized = normalize(matches_flow)

        # Get reference matches and add 1 to z column
        ref_matches = matches[0]
        ref_matches_with_z = np.column_stack((ref_matches, [abs(np.ceil(velocity[2]))]*ref_matches.shape[0]))

        # Transform to camera coordinates
        ref_matches_cam_coord = [inv_k @ x for x in ref_matches_with_z]
        # Inverting z coordinate to match with pixel movement effect when camera translate
        velocity[2] = - velocity[2]

        # Add moved space (just adding velocity: considering time = 1s)
        moved_ref_matches_cam_coord = np.add(ref_matches_cam_coord, velocity)
        # Transform to image coordinates
        moved_ref_matches_img_coord = [k @ x for x in moved_ref_matches_cam_coord]
        # Divide by z to get pixel coordinates
        moved_ref_matches_pixels_coord = np.divide(moved_ref_matches_img_coord, abs(np.ceil(velocity[2])))
        # Removing z coordinate
        moved_ref_matches_pixels_coord = np.delete(moved_ref_matches_pixels_coord, 2, 1)
        # Compute and normalize estimated flow caused by movement
        reference_matches_moved_pixels_coord = np.subtract(moved_ref_matches_pixels_coord, ref_matches)
        reference_matches_moved_image_coord_normalized = normalize(reference_matches_moved_pixels_coord)
        reference_matches_moved_flow = np.subtract(reference_matches_moved_image_coord_normalized, ref_matches)
        reference_matches_moved_flow_normalized = normalize(reference_matches_moved_flow)

        keypoint_diff = np.linalg.norm(matches_flow_normalized - reference_matches_moved_flow_normalized, axis=1)

        inlier_index = np.where(keypoint_diff <= self.inlier_threshold)
        outlier_index = np.where(keypoint_diff > self.inlier_threshold)

        inlier_matches = [matches[0][inlier_index], matches[1][inlier_index]]
        outlier_matches = [matches[0][outlier_index], matches[1][outlier_index]]

        return inlier_matches, outlier_matches

    def get_pose(self, matches):
        pass

    def get_and_update_scale(self, velocity):

        scale = np.linalg.norm(self.prev_velocity - velocity)
        self.prev_velocity = velocity
        return scale


class Visualizer:

    def __init__(self):

        self.trajectory_data = {'ground_truth_x': [], 'ground_truth_y': [], 'ground_truth_z': [],
                                'vo_x':           [], 'vo_y':           [], 'vo_z':           [],
                                'error_x':        [], 'error_y':        [], 'error_z':        []}


    def draw_lines(self, image, start_points, end_points, color):

        for ref_match, curr_match in zip(start_points, end_points):
            image = cv2.arrowedLine(image, tuple(ref_match), tuple(curr_match), color, 1)

        return image

    def update(self, image, inlier_matches, outlier_matches, true_pose, vo_pose, resize_factor):

        # Append true_pose and vo_pose data
        self.trajectory_data['ground_truth_x'].append(true_pose[0])
        self.trajectory_data['ground_truth_y'].append(true_pose[1])
        self.trajectory_data['ground_truth_z'].append(true_pose[2])

        self.trajectory_data['vo_x'].append(vo_pose[0])
        self.trajectory_data['vo_y'].append(vo_pose[1])
        self.trajectory_data['vo_z'].append(vo_pose[2])

        # Compute and append errors
        error_x, error_y, error_z = np.abs(np.subtract(true_pose, vo_pose))
        self.trajectory_data['error_x'].append(error_x)
        self.trajectory_data['error_y'].append(error_y)
        self.trajectory_data['error_z'].append(error_z)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if (inlier_matches is not None) and (outlier_matches is not None):
            image = self.draw_lines(image, inlier_matches[0].astype(int), inlier_matches[1].astype(int), (0, 220, 0))
            image = self.draw_lines(image, outlier_matches[0].astype(int), outlier_matches[1].astype(int), (0, 0, 220))

        h, w = int(image.shape[0] * 1 / resize_factor), int(image.shape[1] * 1 / resize_factor)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imshow('Visual Odometry', image)
        cv2.waitKey(1)

class DataLoader:

    def __init__(self, config: dict):

        # TODO: Implement a validation function to check dataset_config

        # Set Ground Truth and Velocity
        if config['name'] == 'kitti':
            self.resize_factor = config['resize_factor']
            self.dataset = pykitti.raw(config['basedir'], config['date'], config['drive'])
            self.image_getter = iter(self.dataset.cam0)
            self.velocity_getter = ([oxt.packet.vl, oxt.packet.vu, oxt.packet.vf] for oxt in self.dataset.oxts)
            self.true_pose_getter = ([oxt.T_w_imu[0][3], oxt.T_w_imu[1][3], oxt.T_w_imu[2][3]] for oxt in self.dataset.oxts)
            self.dataset.calib_K_cam0 = self.resize_factor * self.dataset.calib.K_cam0; self.dataset.calib_K_cam0[2][2] = 1
            self.dataset.calib_inv_cam0 = np.linalg.inv(self.dataset.calib.K_cam0)

    def get_image(self):

        try:
            image = np.array(next(self.image_getter))
            h, w = int(image.shape[0] * self.resize_factor), int(image.shape[1] * self.resize_factor)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

            return image

        except StopIteration as e:
            raise e

    def get_true_pose(self):

        try:
            true_pose = next(self.true_pose_getter)
            return true_pose
        except StopIteration as e:
            raise e

    def get_velocity(self):

        try:
            velocity = next(self.velocity_getter)
            return np.array(velocity)
        except StopIteration as e:
            raise e


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Monocular Visual Odometry')
    parser.add_argument('--dataset', type=str, help='Dataset id from configuration file', default=None)
    parser.add_argument('--device',  type=str, help='Device: cuda or cpu', default='cuda')
    parser.add_argument('--config',  type=str, help='Path to configuration file', default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f.read())

    # Setup main classes
    data_loader = DataLoader(config['dataset'][args.dataset])
    visualizer = Visualizer()
    visual_odometry = VisualOdometry(config['visual_odometry'], args.device)

    # Get initialization data, initialize visual odometry and update visualizer
    first_image = data_loader.get_image()
    first_velocity = data_loader.get_velocity()
    first_true_pose = data_loader.get_true_pose()

    visual_odometry.set_sg_reference(first_image)
    visual_odometry.prev_velocity = first_velocity
    visualizer.update(first_image, None, None, [0.0, 0.0, 0.0], first_true_pose, data_loader.resize_factor)

    while True:

        try:
            image = data_loader.get_image()
            true_pose = data_loader.get_true_pose()
            velocity = data_loader.get_velocity()

            scale = visual_odometry.get_and_update_scale(velocity)
            if scale > config['visual_odometry']['scale_stop_threshold']:
                matches = visual_odometry.get_matches(image)
                inlier_matches, outlier_matches = visual_odometry.filter_matches(velocity, matches, data_loader.dataset.calib_K_cam0, data_loader.dataset.calib_inv_cam0)
                vo_pose = [1, 1, 1]
                #vo_pose = visual_odometry.get_pose(inlier_matches)
            else:
                inlier_matches, outlier_matches = None, None
                # TODO: Reset inlier_matches and outlier_matches to show that the matching process stopped because the car stopped. vo_pose remains the same

            visualizer.update(image, inlier_matches, outlier_matches, true_pose, vo_pose, data_loader.resize_factor)

        except Exception as e:
            raise e
