import argparse
import cv2
import yaml
import pykitti
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import torch
import sys
from sklearn.preprocessing import normalize
from pyqtgraph.Qt import QtGui, QtCore

from matcher.matching import Matching
from matcher.utils import frame2tensor


class TrackFilter(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super(TrackFilter, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

class VisualOdometry:

    def __init__(self, data_loader, config, device):

        self.device = device
        self.matcher = Matching(config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        self.inlier_threshold = config['inlier_threshold']
        self.R = np.zeros((3, 3))
        self.T = np.zeros((3, 1))

        self.set_prev_velocity(data_loader.get_velocity())
        self.set_sg_reference(data_loader.get_image())

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

    def set_prev_velocity(self, velocity):

        self.prev_velocity = velocity

    def set_sg_reference(self, image):
        """ Set SuperGlue image reference """

        frame_tensor = frame2tensor(image, self.device)
        sg_reference = self.matcher.superpoint({'image': frame_tensor})
        sg_reference = {k + '0': sg_reference[k] for k in self.keys}
        sg_reference['image0'] = frame_tensor
        self.sg_reference = sg_reference

    def filter_matches(self, velocity, matches):

        matches_flow = np.subtract(matches[1], matches[0])
        matches_flow_normalized = normalize(matches_flow)
        velocity = normalize(velocity.reshape(1, -1))[0]

        inlier_index = np.arange(0, 10)
        outlier_index = np.arange(10, 20)

        filtered_matches = {
            'inlier': [matches[0][inlier_index], matches[1][inlier_index]],
            'outlier': [matches[0][outlier_index], matches[1][outlier_index]]
        }

        return filtered_matches

    def get_pose(self, matches, focal, pp, scale):

        E, mask = cv2.findEssentialMat(matches['reference'], matches['target'], focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=0.5)
        _, R, T, mask = cv2.recoverPose(E, matches['reference'], matches['target'], focal=focal, pp=pp)

        self.T = self.T + scale * self.R.dot(T)
        self.R = self.R.dot(R)

        return self.T

    def get_and_update_scale(self, velocity):

        scale = np.linalg.norm(self.prev_velocity - velocity)
        self.prev_velocity = velocity
        return scale

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
            self.k = self.dataset.k = self.resize_factor * self.dataset.calib.K_cam0; self.k[2][2] = 1
            self.inv_k = np.linalg.inv(self.k)
            self.focal = self.k[0][0]
            self.pp = (self.k[0][2], self.k[1][2])

    def get_image(self):

        try:
            image = np.array(next(self.image_getter))
            resized_h, resized_w = int(image.shape[0] * self.resize_factor), int(image.shape[1] * self.resize_factor)
            image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

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

class Commander(object):

    def __init__(self, args):

        with open(args.config, 'r') as f:
            self.config = yaml.safe_load(f.read())

        self.device = args.device
        self.data_loader = DataLoader(self.config['dataset'][args.dataset])
        self.visual_odometry = VisualOdometry(self.data_loader, self.config['visual_odometry'], args.device)

        # PyQtGraph
        ## App
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)

        ## 2D View
        self.view_2d = pg.GraphicsWindow(title='VIO - Trajectory Error and Matches')

        ## 3D View
        self.view_3d = gl.GLViewWidget()
        self.view_3d.setGeometry(5, 115, 1200, 800)
        self.view_3d.opts['distance'] = 200
        self.view_3d.setWindowTitle('VIO - Trajectory')
        gx, gy, gz = gl.GLGridItem(), gl.GLGridItem(), gl.GLGridItem()
        gx.setSpacing(2, 2); gy.setSpacing(2, 2); gz.setSpacing(2, 2)
        gx.setSize(70, 70); gy.setSize(70, 70); gz.setSize(70, 70);
        gx.rotate(90, 0, 1, 0); gx.translate(-35, 0, 0);
        gy.rotate(90, 1, 0, 0); gy.translate(0, -35, 0)
        gz.translate(0, 0, -35);
        self.view_3d.addItem(gx); self.view_3d.addItem(gy); self.view_3d.addItem(gz)
        self.view_3d.show()


        ## Plots
        self.plot_data = {'error': {'x': [], 'y': [], 'z': []},
                          'true_pose': [],
                          'vo_pose': []}

        self.trajectory_error_plot = self.view_2d.addPlot(title='Trajectory Error', row=1, col=1, labels={'left': 'Error (m)', 'bottom': 'Frame ID'})

        self.plots = {'error': {'x': self.trajectory_error_plot.plot(pen=(255, 0, 0), width=3, name='Error x'),
                                'y': self.trajectory_error_plot.plot(pen=(0, 255, 0), width=3, name='Error y'),
                                'z': self.trajectory_error_plot.plot(pen=(0, 0, 255), width=3, name='Error z')},
                      'true_pose': gl.GLScatterPlotItem(color=(0.0, 0.0, 1.0, 1), size=1, pxMode=False),
                      'vo_pose': gl.GLScatterPlotItem(color=(0.0, 1.0, 0.0, 1), size=1, pxMode=False)}

        self.view_3d.addItem(self.plots['true_pose'])
        self.view_3d.addItem(self.plots['vo_pose'])
        self.sequence_id = 0

    def update(self):

        self.sequence_id += 1
        self.image = self.data_loader.get_image()
        self.pose = {'true_pose': self.data_loader.get_true_pose(), 'vo_pose': [0, 0, 0]}
        self.velocity = self.data_loader.get_velocity()

        # scale = visual_odometry.get_and_update_scale(velocity)
        # if scale > config['visual_odometry']['scale_stop_threshold']:
        #     matches = visual_odometry.get_matches(image)
        #     filtered_matches = visual_odometry.filter_matches(velocity, matches)
        #     pose['vo_pose'] = visual_odometry.get_pose(inlier_matches, focal=data_loader.focal, pp=data_loader.pp)
        # else:
        #     inlier_matches, outlier_matches = None, None
        #     # TODO: Reset inlier_matches and outlier_matches to show that the matching process stopped because the car stopped. vo_pose remains the same

        # Append new data
        ## Trajectory
        self.plot_data['true_pose'].append([self.pose['true_pose'][0], self.pose['true_pose'][1], self.pose['true_pose'][2]])
        self.plot_data['vo_pose'].append([self.pose['vo_pose'][0], self.pose['vo_pose'][1], self.pose['vo_pose'][2]])


        ## Trajectory Error
        error = abs(np.subtract(self.pose['true_pose'], self.pose['vo_pose']))
        self.plot_data['error']['x'].append(error[0])
        self.plot_data['error']['y'].append(error[1])
        self.plot_data['error']['z'].append(error[2])

        # Update plots
        ## Trajectory
        self.plots['true_pose'].setData(pos=np.array(self.plot_data['true_pose']))
        self.plots['vo_pose'].setData(pos=np.array(self.plot_data['vo_pose']))

        ## Trajectory Error
        self.plots['error']['x'].setData(x=np.arange(self.sequence_id), y=self.plot_data['error']['x'])
        self.plots['error']['y'].setData(x=np.arange(self.sequence_id), y=self.plot_data['error']['y'])
        self.plots['error']['z'].setData(x=np.arange(self.sequence_id), y=self.plot_data['error']['z'])


    def start(self):

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Monocular Visual Odometry')
    parser.add_argument('--dataset', type=str, help='Dataset id from configuration file')
    parser.add_argument('--config',  type=str, help='Path to configuration file', default='config.yaml')
    parser.add_argument('--device',  type=str, help='Device: cuda or cpu', default='cuda')
    args = parser.parse_args()

    commander = Commander(args)
    commander.start()
















        # class VisualizerOld:
        #
        #     def __init__(self):
        #
        #         self.trajectory_data = {'ground_truth_x': [], 'ground_truth_y': [], 'ground_truth_z': [],
        #                                 'vo_x': [], 'vo_y': [], 'vo_z': [],
        #                                 'error_x': [], 'error_y': [], 'error_z': []}
        #
        #     def draw_lines(self, image, start_points, end_points, color):
        #
        #         for ref_match, curr_match in zip(start_points, end_points):
        #             image = cv2.arrowedLine(image, tuple(ref_match), tuple(curr_match), color, 1)
        #
        #         return image
        #
        #     def update(self, image, inlier_matches, outlier_matches, true_pose, vo_pose, resize_factor):
        #
        #         # Append true_pose and vo_pose data
        #         self.trajectory_data['ground_truth_x'].append(true_pose[0])
        #         self.trajectory_data['ground_truth_y'].append(true_pose[1])
        #         self.trajectory_data['ground_truth_z'].append(true_pose[2])
        #
        #         self.trajectory_data['vo_x'].append(vo_pose[0])
        #         self.trajectory_data['vo_y'].append(vo_pose[1])
        #         self.trajectory_data['vo_z'].append(vo_pose[2])
        #
        #         # Compute and append errors
        #         error_x, error_y, error_z = np.abs(np.subtract(true_pose, vo_pose))
        #         self.trajectory_data['error_x'].append(error_x)
        #         self.trajectory_data['error_y'].append(error_y)
        #         self.trajectory_data['error_z'].append(error_z)
        #
        #         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #         if (inlier_matches is not None) and (outlier_matches is not None):
        #             image = self.draw_lines(image, inlier_matches[0].astype(int), inlier_matches[1].astype(int),
        #                                     (0, 220, 0))
        #             image = self.draw_lines(image, outlier_matches[0].astype(int), outlier_matches[1].astype(int),
        #                                     (0, 0, 220))
        #
        #         h, w = int(image.shape[0] * 1 / resize_factor), int(image.shape[1] * 1 / resize_factor)
        #         image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        #         cv2.imshow('Visual Odometry', image)
        #         cv2.waitKey(1)

        # # Get reference matches and add 1 to z column
        # ref_matches = matches[0]
        # ref_matches_with_z = np.column_stack((ref_matches, [abs(np.ceil(velocity[2]))]*ref_matches.shape[0]))
        #
        # # Transform to camera coordinates
        # ref_matches_cam_coord = [inv_k @ x for x in ref_matches_with_z]
        # # Inverting z coordinate to match with pixel movement effect when camera translate
        # velocity[2] = - velocity[2]
        #
        # # Add moved space (just adding velocity: considering time = 1s)
        # moved_ref_matches_cam_coord = np.add(ref_matches_cam_coord, velocity)
        # # Transform to image coordinates
        # moved_ref_matches_img_coord = [k @ x for x in moved_ref_matches_cam_coord]
        # # Divide by z to get pixel coordinates
        # moved_ref_matches_pixels_coord = np.divide(moved_ref_matches_img_coord, abs(np.ceil(velocity[2])))
        # # Removing z coordinate
        # moved_ref_matches_pixels_coord = np.delete(moved_ref_matches_pixels_coord, 2, 1)
        # # Compute and normalize estimated flow caused by movement
        # reference_matches_moved_pixels_coord = np.subtract(moved_ref_matches_pixels_coord, ref_matches)
        # reference_matches_moved_image_coord_normalized = normalize(reference_matches_moved_pixels_coord)
        # reference_matches_moved_flow = np.subtract(reference_matches_moved_image_coord_normalized, ref_matches)
        # reference_matches_moved_flow_normalized = normalize(reference_matches_moved_flow)
        #
        # keypoint_diff = np.linalg.norm(matches_flow_normalized - reference_matches_moved_flow_normalized, axis=1)

        #        inlier_index = np.where(keypoint_diff <= self.inlier_threshold)
        #outlier_index = np.where(keypoint_diff > self.inlier_threshold)
        # inlier_matches = [matches[0][inlier_index], matches[1][inlier_index]]
        # outlier_matches = [matches[0][outlier_index], matches[1][outlier_index]]