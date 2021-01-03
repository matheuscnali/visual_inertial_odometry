import argparse
import cv2
import yaml
import pykitti
import numpy as np
import pyqtgraph.opengl as gl
import torch
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
from models import matcher, utils
from motion_detection import motion_probability
from models.raft.raft import RAFT


def draw_lines(image, points, color):

    for ref_match, cur_match in zip(points['ref_keypoints'], points['cur_keypoints']):
        image = cv2.arrowedLine(image, tuple(ref_match.astype(np.int16)), tuple(cur_match.astype(np.int16)), color, 1)

    return image

class VisualOdometry:

    def __init__(self, args, data_loader, config):

        self.matching = matcher.Matching(config['matcher']).eval().to(device='cuda')

        self.prev_image = data_loader.get_image()
        self.cur_image = data_loader.get_image()

        self.prev_image['left_cuda'] = utils.frame2tensor(self.prev_image['left_gray'], 'cuda')
        self.cur_image['left_cuda'] = utils.frame2tensor(self.cur_image['left_gray'], 'cuda')

        self.matches = self.matching({'image0': self.prev_image['left_cuda'], 'image1': self.cur_image['left_cuda']})

        self.inlier_threshold = config['inlier_threshold']
        self.prev_pose = data_loader.get_true_pose()
        self.pose = {'true_pose': [self.prev_pose], 'vo_pose': [[0, 0, 0]]}

        self.R, self.T = np.identity(3), np.zeros((3, 1))

        self.optical_flow_model = torch.nn.DataParallel(RAFT(args))
        self.optical_flow_model.load_state_dict(torch.load(args.optical_flow_model))
        model = self.optical_flow_model.module
        model.to('cuda').eval()

        y, x = np.mgrid[0: self.prev_image['left_gray'].shape[0]: 1, 0: self.prev_image['left_gray'].shape[1]: 1].reshape(2, -1).astype(int)
        self.dynamic_model = np.array([x ** 2, y ** 2, x * y, x, y, x * 0 + 1]).T
        config['motion_detection']['x'], config['motion_detection']['y'] = x, y
        config['motion_detection']['height'], config['motion_detection']['width'] = self.prev_image['left_gray'].shape[0], self.prev_image['left_gray'].shape[1]

        self.prev_image.update(**self.cur_image)

    def get_matches(self, image):

        self.cur_image['left_cuda'] = utils.frame2tensor(image, 'cuda')

        self.matches = self.matching({'image0': self.prev_image['left_cuda'], 'image1': self.cur_image['left_cuda']})

        confidence = self.matches['matching_scores0'][0].detach().cpu().numpy()
        valid_match = self.matches['matches0'][0].cpu().numpy() > -1
        valid_confidence = confidence > 0.85
        valid = np.logical_and(valid_match, valid_confidence)
        self.mkpts0 = self.matches['keypoints0'][0].detach().cpu().numpy()[valid]
        self.mkpts1 = self.matches['keypoints1'][0].detach().cpu().numpy()[self.matches['matches0'][0].cpu().numpy()[valid]]

        return {'ref_keypoints': self.mkpts0, 'cur_keypoints': self.mkpts1}

    def filter_matches(self, bg_mask, matches):

        inliers, outliers = {'ref_keypoints': [], 'cur_keypoints': []}, {'ref_keypoints': [], 'cur_keypoints': []}

        for ref_keypoint, cur_keypoint in zip(matches['ref_keypoints'], matches['cur_keypoints']):
            if bg_mask[int(ref_keypoint[1])][int(ref_keypoint[0])]:
                inliers['ref_keypoints'].append(ref_keypoint)
                inliers['cur_keypoints'].append(cur_keypoint)
            else:
                outliers['ref_keypoints'].append(ref_keypoint)
                outliers['cur_keypoints'].append(cur_keypoint)

        inliers['ref_keypoints'], inliers['cur_keypoints'] = np.array(inliers['ref_keypoints']), np.array(inliers['cur_keypoints'])
        outliers['ref_keypoints'], outliers['cur_keypoints'] = np.array(outliers['ref_keypoints']), np.array(outliers['cur_keypoints'])

        return {'inlier': inliers, 'outlier': outliers}

    def get_pose(self, matches, focal, pp, scale):

        E, mask = cv2.findEssentialMat(matches['cur_keypoints'], matches['ref_keypoints'], focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, T, mask = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'], focal=focal, pp=pp)

        self.T = self.T + scale * self.R @ T
        self.R = self.R @ R

        return self.T.flatten()

    def get_and_update_scale(self, pose):

        scale = np.linalg.norm(np.subtract(self.prev_pose, pose))
        self.prev_pose = pose
        return scale

class DataLoader:

    def __init__(self, config: dict):

        # Set Ground Truth and Velocity
        if config['name'] == 'kitti_raw':
            self.resize_factor = config['resize_factor']
            self.dataset = pykitti.raw(config['basedir'], config['date'], config['drive'])
            self.image_getter_left = iter(self.dataset.cam0)
            self.velocity_getter = ([oxt.packet.vl, oxt.packet.vu, oxt.packet.vf] for oxt in self.dataset.oxts)
            self.true_pose_getter = ([oxt.T_w_imu[0][3], oxt.T_w_imu[1][3], oxt.T_w_imu[2][3]] for oxt in self.dataset.oxts)
            self.k = self.dataset.k = self.resize_factor * self.dataset.calib.K_cam0
            self.k[2][2] = 1
            self.inv_k = np.linalg.inv(self.k)
            self.focal = self.k[0][0]
            self.pp = (self.k[0][2], self.k[1][2])

        if config['name'] == 'kitti_odometry':
            self.resize_factor = config['resize_factor']
            self.dataset = pykitti.odometry(config['basedir'], config['sequence'])
            self.image_getter_left = iter(self.dataset.cam0)
            self.image_getter_right = iter(self.dataset.cam1)
            self.true_pose_getter = ([pose[0][3], pose[1][3], pose[2][3]] for pose in self.dataset.poses)
            self.k = self.dataset.k = self.resize_factor * self.dataset.calib.K_cam0
            self.k[2][2] = 1
            self.inv_k = np.linalg.inv(self.k)
            self.focal = self.k[0][0]
            self.pp = (self.k[0][2], self.k[1][2])

    def get_image(self):

        try:
            left_image_gray = np.array(next(self.image_getter_left))
            if left_image_gray.ndim == 3:
                left_image_gray = cv2.cvtColor(left_image_gray, cv2.COLOR_RGB2GRAY)
            resized_h, resized_w = int(left_image_gray.shape[0] * self.resize_factor), int(left_image_gray.shape[1] * self.resize_factor)
            left_image_gray = cv2.resize(left_image_gray, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

            return {'left_gray': left_image_gray}

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
            if 'velocity_getter' in dir(self):
                velocity = next(self.velocity_getter)
                return np.array(velocity)
        except StopIteration as e:
            raise e

class Commander(object):

    def setup_pyqtgraph(self):
        ## App
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)

        ## 2D View
        self.view_2d = pg.GraphicsWindow(title='VIO - Trajectory Error and Matches')

        ## 3D View
        self.view_3d = gl.GLViewWidget()
        size = 30000
        self.view_3d.setGeometry(5, 115, size, size)
        self.view_3d.opts['distance'] = size/5
        self.view_3d.setWindowTitle('VIO - Trajectory')
        gx, gy, gz = gl.GLGridItem(), gl.GLGridItem(), gl.GLGridItem()
        gx.setSpacing(size/100, size/100); gy.setSpacing(size/100, size/100); gz.setSpacing(size/100, size/100)
        gx.setSize(size, size); gy.setSize(size, size); gz.setSize(size, size);
        gx.rotate(90, 0, 1, 0); gx.translate(-size//2, 0, 0);
        gy.rotate(90, 1, 0, 0); gy.translate(0, -size//2, 0)
        gz.translate(0, 0, -size//2);
        self.view_3d.addItem(gx); self.view_3d.addItem(gy); self.view_3d.addItem(gz)
        self.view_3d.show()

        ## Matches View
        self.matches_win = pg.GraphicsLayoutWidget()
        self.matches_win.setWindowTitle('Inliers and outliers matches')

        ## A plot area (ViewBox + axes) for displaying the image
        self.match_plot = self.matches_win.addPlot(title="Inliers and outliers matches")

        ## Item for displaying image data
        self.matches_img = pg.ImageItem()
        self.match_plot.addItem(self.matches_img)
        self.matches_win.show()

        ## Plots
        self.plot_data = {'error': {'x': [], 'y': [], 'z': []},
                          'true_pose': [],
                          'vo_pose': []}

        self.trajectory_error_plot = self.view_2d.addPlot(title='Trajectory Error', row=1, col=1, labels={'left': 'Error (m)', 'bottom': 'Frame ID'})

        self.trajectory_error_plot.addLegend()
        self.plots = {'error': {'x': self.trajectory_error_plot.plot(pen=(255, 0, 0), width=3, name='Error x'),
                                'y': self.trajectory_error_plot.plot(pen=(0, 255, 0), width=3, name='Error y'),
                                'z': self.trajectory_error_plot.plot(pen=(0, 0, 255), width=3, name='Error z')},
                      'true_pose': gl.GLScatterPlotItem(color=(0.0, 0.0, 1.0, 1), size=1, pxMode=False), # Blue
                      'vo_pose': gl.GLScatterPlotItem(color=(0.0, 1.0, 0.0, 1), size=1, pxMode=False)} # Green

        self.view_3d.addItem(self.plots['true_pose'])
        self.view_3d.addItem(self.plots['vo_pose'])
        self.sequence_id = 0

    def __init__(self, args):

        with open(args.config, 'r') as f:
            self.config = yaml.safe_load(f.read())

        self.data_loader = DataLoader(self.config['dataset'][args.dataset])
        self.visual_odometry = VisualOdometry(args, self.data_loader, self.config['visual_odometry'])
        self.setup_pyqtgraph()

    def update_view(self):

        # Append new data
        ## Trajectory
        self.plot_data['true_pose'].append([*self.visual_odometry.pose['true_pose']])
        self.plot_data['vo_pose'].append([*self.visual_odometry.pose['vo_pose']])

        ## Trajectory Error
        error = abs(np.subtract(self.visual_odometry.pose['true_pose'], self.visual_odometry.pose['vo_pose']))
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

        ## Matches
        self.image = cv2.cvtColor(self.visual_odometry.cur_image['left_gray'], cv2.COLOR_GRAY2RGB)
        self.image = draw_lines(self.image, self.filtered_matches['inlier'], color=(0, 255, 0))
        self.image = draw_lines(self.image, self.filtered_matches['outlier'], color=(255, 0, 0))
        self.matches_img.setImage(np.rot90(self.image, axes=(1, 0)))

    def step(self):

        try:
            self.sequence_id += 1
            self.visual_odometry.cur_image['left_gray'] = self.data_loader.get_image()['left_gray']
            self.visual_odometry.pose['true_pose'] = self.data_loader.get_true_pose()

            self.scale = self.visual_odometry.get_and_update_scale(self.visual_odometry.pose['true_pose'])
            if self.scale > self.config['visual_odometry']['scale_stop_threshold']:
                self.matches = self.visual_odometry.get_matches(self.visual_odometry.cur_image['left_gray'])
                self.bg_mask = motion_probability(self.visual_odometry.prev_image['left_gray'], self.visual_odometry.cur_image['left_gray'], self.visual_odometry.optical_flow_model, self.visual_odometry.dynamic_model, self.config['visual_odometry']['motion_detection'])
                self.filtered_matches = self.visual_odometry.filter_matches(self.bg_mask, self.matches)
                self.visual_odometry.pose['vo_pose'] = self.visual_odometry.get_pose(self.filtered_matches['inlier'], focal=self.data_loader.focal, pp=self.data_loader.pp, scale=self.scale)
            else:
                self.filtered_matches = {'inlier': None, 'outlier': None}

            self.update_view()
            self.visual_odometry.prev_image.update(**self.visual_odometry.cur_image)

        except StopIteration as e:
            commander.view_3d.grabFrameBuffer().save('trajectory.png')
            exporter = pg.exporters.ImageExporter(self.trajectory_error_plot)
            exporter.export('trajectory_error.png')
            print(f"Last error x: {self.plot_data['error']['x'][-1]}")
            print(f"Last error y: {self.plot_data['error']['y'][-1]}")
            print(f"Last error z: {self.plot_data['error']['z'][-1]}")

    def start(self):

        timer = QtCore.QTimer()
        timer.timeout.connect(self.step)
        timer.start(20)

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Monocular Visual Odometry')
    parser.add_argument('--dataset', type=str, help='Dataset id from configuration file')
    parser.add_argument('--config',  type=str, help='Path to configuration file', default='config.yaml')
    parser.add_argument('--optical_flow_model', help="restore checkpoint", default='models/raft/raft-things.pth', )
    parser.add_argument('--small', help='use small model for optical flow', action='store_true', )
    parser.add_argument('--mixed_precision', help='use mixed precision', action='store_true', )
    parser.add_argument('--alternate_corr', help='use efficient correlation implementation', action='store_true', )
    args = parser.parse_args()

    commander = Commander(args)
    commander.start()
