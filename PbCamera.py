import copy
import json
import math
from typing import Dict, List, Tuple
import numpy as np
from pandas import array
import pybullet as pb
import open3d as o3d
import pybullet_data
import cv2
import os
from math import cos, sin
import torch
from treelib import Node, Tree
from collections import OrderedDict
import random
from PbJoint import PbJoint, PbJointAxis, PbLink
from transform import Transform, get_transform
from dataclasses import dataclass, asdict

from pybullet_utils import bullet_client

class PbCamera():
    def __init__(self, pixelWidth, pixelHeight, connection_mode=None) -> None:
        self.upAxisIndex = 2    # 1: Y up; 2: Z up.
        self.camDistance = 2.5
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
        self.nearPlane = 0.01
        self.farPlane = 100
        self.fov = 60
        self.camTargetPos = [0, 0, 0]
        self.roll = 0   # x, lock to 0
        self.pitch = 0  # y
        self.yaw = 0    # z
        self.physicsClient = bullet_client.BulletClient(connection_mode)
        self.physicsClient.setPhysicsEngineParameter(enableFileCaching=0)
        self.o3d_viewer = None
        self.vx = 0
        self.vy = 0
        self.calculate_matrix()
        self.paramStorage = {
            "projectionMatrixes": [],
            "viewMatrixes": [],
            "invPVMatrixes": [],
        }

    def calculate_matrix(self):
        '''
        POSITION_IN_IMAGE = PROJECTION_MATRIX * POSE_IN_CAMERA_SPACE
        POSE_IN_CAMERA_SPACE = VIEW_MATRIX * POSE_IN_WORLD_SPACE
        POSE_IN_WORLD_SPACE = MODEL_MATRIX * POSE_IN_MODEL_SPACE
        '''
        self.viewMatrix = self.physicsClient.computeViewMatrixFromYawPitchRoll(self.camTargetPos, self.camDistance, self.yaw, self.pitch,
                                                                               self.roll, self.upAxisIndex)
        aspect = self.pixelWidth / self.pixelHeight
        self.projectionMatrix = self.physicsClient.computeProjectionMatrixFOV(
            self.fov, aspect, self.nearPlane, self.farPlane)

        self.projectionMatrix = np.asarray(
            self.projectionMatrix).reshape([4, 4], order="F")
        self.viewMatrix = np.asarray(
            self.viewMatrix).reshape([4, 4], order="F")
        self.invPVMatrix = np.linalg.inv(
            np.matmul(self.projectionMatrix, self.viewMatrix))

    def set_rpy(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.calculate_matrix()

    def set_pixelsize(self, w, h):
        self.pixelHeight = h
        self.pixelWidth = w
        self.calculate_matrix()

    def set_visualarea(self, fov, near, far):
        self.fov = fov
        self.nearPlane = near
        self.farPlane = far
        self.calculate_matrix()

    def set_focus(self, target_pos):
        self.camTargetPos = target_pos
        self.calculate_matrix()

    def set_distance(self, dis):
        self.camDistance = dis
        self.calculate_matrix()

    def get_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        gl_view_matrix = self.viewMatrix.flatten(order="F")
        gl_proj_matrix = self.projectionMatrix.flatten(order="F")
        img_arr = self.physicsClient.getCameraImage(width=self.pixelWidth, height=self.pixelHeight, viewMatrix=gl_view_matrix,
                                                    projectionMatrix=gl_proj_matrix, renderer=self.physicsClient.ER_TINY_RENDERER, flags=self.physicsClient.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)

        if type(img_arr[2]) == tuple:
            img_rgb = np.reshape(np.array(img_arr[2]), (self.pixelHeight, self.pixelWidth, 4))
        else:
            img_rgb = img_arr[2]
        if type(img_arr[3]) == tuple:
            img_depth = np.reshape(np.array(img_arr[3]), (self.pixelHeight, self.pixelWidth))
        else:
            img_depth = img_arr[3]
        if type(img_arr[4]) == tuple:
            img_mask = np.reshape(np.array(img_arr[4]), (self.pixelHeight, self.pixelWidth))
        else:
            img_mask = img_arr[4]

        return img_rgb, img_depth, img_mask
    
    def uniform_camera_pose(self, r_range: Tuple, p_range: Tuple, y_range: Tuple):
        rpy_list = []
        for roll in range(*r_range):
            for pitch in range(*p_range):
                for yaw in range(*y_range):
                    rpy_list.append([roll, pitch, yaw])
        return rpy_list

    def get_full_pointcloud(self, point_num=10000):

        rpy_list = []
        for pitch in range(0, 180, 30):
            for yaw in range(0, 180, 30):
                rpy_list.append((90, pitch, yaw))
                # rpy_list.append((0, 60, 90))

        points = []
        for rpy in rpy_list:
            self.set_rpy(rpy[0], rpy[1], rpy[2])
            self.physicsClient.stepSimulation()
            img_rgb, img_depth, img_mask = self.get_image()
            points.append(self.create_pointcloud(img_depth, img_mask, img_rgb))

        pointcloud = torch.from_numpy(np.concatenate(points)).to("cuda:0")
        idx = self.farthest_point_sample(pointcloud[:, :3], point_num=point_num)
        pointcloud = pointcloud[idx].cpu().numpy()

        return pointcloud, points, rpy_list

    def get_single_pointcloud(self, rpy: tuple):
        self.set_rpy(rpy[0], rpy[1], rpy[2])
        self.physicsClient.stepSimulation()
        img_rgb, img_depth, img_mask = self.get_image()
        pointcloud = self.create_pointcloud(img_depth, img_mask, img_rgb)

        return pointcloud

    def create_pointcloud(self, img_depth, img_mask=np.array([]), img_rgb=np.array([])):

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:self.pixelHeight*1j, -1:1:self.pixelWidth*1j]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), 2 * \
            img_depth.reshape(-1)-1

        h = np.ones_like(z)
        pixels = np.stack([x, y, z, h], axis=1)

        # filter out "infinite" depths
        filter_mask = z < 0.999999999
        pixels = pixels[filter_mask]

        # turn pixels to world coordinates
        points = np.matmul(self.invPVMatrix, pixels.T).T
        points /= points[:, 3:]
        points = points[:, :3]

        points_with_label = points

        if not img_mask.shape[0] == 0:

            seg = img_mask.reshape(-1)
            seg_shape = seg.shape
            non_neg_seg_idx = seg >= 0
            obj_uid = - np.ones_like(seg)
            link_index = - 2 * np.ones_like(seg)
            obj_uid[non_neg_seg_idx] = seg[non_neg_seg_idx] & ((1 << 24) - 1)
            link_index[non_neg_seg_idx] = (seg[non_neg_seg_idx] >> 24) - 1
            obj_uid = obj_uid.reshape(seg_shape)
            link_index = link_index.reshape(seg_shape)

            obj_uid = obj_uid[filter_mask]
            link_index = link_index[filter_mask]

            points_with_label = np.column_stack(
                [points_with_label, obj_uid, link_index])

        if not img_rgb.shape[0] == 0:
            r, g, b = img_rgb[:, :, 0].reshape(-1), img_rgb[:,
                                                            :, 1].reshape(-1), img_rgb[:, :, 2].reshape(-1)
            r = r[filter_mask]
            g = g[filter_mask]
            b = b[filter_mask]

            points_with_label = np.column_stack(
                [points_with_label, r, g, b])

        return points_with_label

    def save_pointcloud_as_npy(self, pointcloud, save_path="./", prefix=None):
        save_name = os.path.join(save_path, "{}_pc.npy".format(prefix))
        np.save(save_name, pointcloud)

    def seg_to_color(self, seg: np.ndarray) -> np.ndarray:
        label_to_color = {
            0: [255, 0, 255],
            1: [0, 0, 255],
            2: [0, 255, 255],
            3: [0, 255, 0],
            4: [255, 255, 0],
            5: [255, 0, 0]
        }
        N = len(seg)
        color = np.zeros((N, 3))
        for i in range(N):
            color[i] = label_to_color[seg[i] % len(label_to_color)]
        return color

    def save_pointcloud_as_ply(self, pointcloud, save_path="./", use_rgb=False, prefix=None,):
        save_name = os.path.join(save_path, "{}_pc.ply".format(prefix))
        num_points = pointcloud.shape[0]
        points = pointcloud[:, 0:3]
        if use_rgb:
            colors = self.seg_to_color(pointcloud[:, 4])
            # Save the point cloud in the PLY format
            with open(save_name, 'w') as f:
                # Write the header
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write(f'element vertex {num_points}\n')
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
                f.write('end_header\n')

                # Write the point data
                for i in range(num_points):
                    f.write(f'{points[i,0]} {points[i,1]} {points[i,2]} ')
                    f.write(
                        f'{int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n')
        else:
            with open(save_name, 'w') as f:
                # Write the header
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write(f'element vertex {num_points}\n')
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('end_header\n')

                # Write the point data
                for i in range(num_points):
                    f.write(f'{points[i,0]} {points[i,1]} {points[i,2]} ')

    def load_pointcloud(self):
        raise NotImplementedError

    def _get_cross_prod_mat(self, pVec_Arr):
        # pVec_Arr shape (3)
        qCross_prod_mat = np.array([
            [0, -pVec_Arr[2], pVec_Arr[1]],
            [pVec_Arr[2], 0, -pVec_Arr[0]],
            [-pVec_Arr[1], pVec_Arr[0], 0],
        ])
        return qCross_prod_mat

    def _caculate_align_mat(self, pVec_Arr):
        scale = np.linalg.norm(pVec_Arr)
        pVec_Arr = pVec_Arr / scale
        # must ensure pVec_Arr is also a unit vec.
        z_unit_Arr = np.array([0, 0, 1])
        z_mat = self._get_cross_prod_mat(z_unit_Arr)

        z_c_vec = np.matmul(z_mat, pVec_Arr)
        z_c_vec_mat = self._get_cross_prod_mat(z_c_vec)

        if np.dot(z_unit_Arr, pVec_Arr) == -1:
            qTrans_Mat = -np.eye(3, 3)
        elif np.dot(z_unit_Arr, pVec_Arr) == 1:
            qTrans_Mat = np.eye(3, 3)
        else:
            qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                                z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))

        qTrans_Mat *= scale
        return qTrans_Mat

    def _apply_viewmatrix(self, pos):
        pos = np.append(pos, 1)
        pos = (np.matmul(pos, self.viewMatrix.T).T)
        pos /= pos[3]
        pos = pos[:3]
        return pos

    def get_mesh_arrow(self, start_pos: Tuple[float, float, float] = (.0, .0, .0),
                       direction: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                       color: Tuple[int, int, int] = (1, 0.5, 0),
                       cone_height: float = 0.08,
                       cone_radius: float = 0.03,
                       cylinder_height: float = 0.8,
                       cylinder_radius: float = 0.02,
                       ):

        mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=cone_height,
            cone_radius=cone_radius,
            cylinder_height=cylinder_height,
            cylinder_radius=cylinder_radius
        )
        mesh_arrow.paint_uniform_color(color)
        mesh_arrow.compute_vertex_normals()

        rot_mat = self._caculate_align_mat(direction)
        mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
        mesh_arrow.translate(np.array(start_pos))

        return mesh_arrow

    def draw_pointcloud(self, points: np.ndarray, zoom=0.3412,
                        front=[0.4257, -0.2125, -0.8795],
                        lookat=[0, 0, 0],
                        up=[-0.0694, -0.9768, 0.2024],
                        joints: List[PbJointAxis] = []
                        ) -> None:
        # print("range: ", points.max(axis=0), points.min(axis=0))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
        mesh_arrow_list = []
        for joint in joints:
            mesh_arrow_list.append(self.get_mesh_arrow(
                start_pos=joint.globalPos, direction=joint.globalAxis))
        o3d.visualization.draw_geometries(
            [FOR1, pcd]+mesh_arrow_list, zoom=zoom, front=front, lookat=lookat, up=up)

    def show_pointcloud(self, points: np.ndarray, zoom=0.3412,
                        front=[0.4257, -0.2125, -0.8795],
                        lookat=[0, 0, 0],
                        up=[-0.0694, -0.9768, 0.2024],
                        joints: List[PbJointAxis] = []
                        ) -> None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
        mesh_arrow_list = []
        for joint in joints:
            mesh_arrow_list.append(self.get_mesh_arrow(
                start_pos=joint.globalPos, direction=joint.globalAxis))

        if not self.o3d_viewer:
            self.o3d_viewer = o3d.visualization.Visualizer()
            self.o3d_viewer.create_window()
        self.o3d_viewer.clear_geometries()
        self.o3d_viewer.add_geometry(pcd)
        self.o3d_viewer.add_geometry(FOR1)
        for joint in joints:
            self.o3d_viewer.add_geometry(self.get_mesh_arrow(
                start_pos=joint.globalPos, direction=joint.globalAxis))

        ctr = self.o3d_viewer.get_view_control()
        # self.vx+=10
        # self.vy-=10
        # print("vx: {} vy: {}".format(self.vx, self.vy))
        ctr.rotate(5410, -5410, 0, 0)

        self.o3d_viewer.poll_events()
        self.o3d_viewer.update_renderer()

    def _normalize_pointcloud(self, points):
        raise NotImplementedError
    
    def uniform_sample(self, alpha, N):
        """
        Returns N uniformly distributed random numbers between 0 and alpha,
        sampled from N equally spaced intervals within the interval [0, alpha].
        """
        interval_size = alpha / N
        interval_starts = np.arange(0, alpha, interval_size)
        samples = np.random.uniform(interval_starts, interval_starts + interval_size, N)
        return samples

    def generate_state_lists(self, joint_list: List[PbJointAxis], object_type: str, state_num: int) -> np.ndarray:
        state_lists = np.zeros(shape=(len(joint_list), state_num))

        for i in range(len(joint_list)):
            jointMiddleState = (
                joint_list[i].jointUpperLimit + joint_list[i].jointLowerLimit)/2
            jointMoveRange = joint_list[i].jointUpperLimit - \
                joint_list[i].jointLowerLimit
            if joint_list[i].jointType == pb.JOINT_REVOLUTE:
                if object_type == 'eyeglasses':
                    state_lists[i] = self.uniform_sample(np.pi/2, state_num)
                    # state_lists[i] = np.array(
                    #     [(90/state_num*(i+np.random.random()-0.5)) for i in range(state_num)])*math.pi/180
                else:
                    state_lists[i] = self.uniform_sample(np.pi, state_num)
                    # state_lists[i] = np.array(
                    #     [(180/state_num*(i+np.random.random()-0.5)) for i in range(state_num)])*math.pi/180
            elif joint_list[i].jointType == pb.JOINT_PRISMATIC:
                state_lists[i] = jointMiddleState + \
                    np.array([0.025*i for i in range(state_num)])*jointMoveRange
            np.random.shuffle(state_lists[i])
        return state_lists.T

    def generate_middle_state(self, joint_list: List[PbJointAxis]) -> np.ndarray:
        state = np.zeros(shape=(len(joint_list)))
        for i in range(len(joint_list)):
            if joint_list[i].jointType == pb.JOINT_REVOLUTE:
                state[i] = math.pi/4
            elif joint_list[i].jointType == pb.JOINT_PRISMATIC:
                state[i] = joint_list[i].jointLowerLimit+0.75 * \
                    (joint_list[i].jointUpperLimit -
                     joint_list[i].jointLowerLimit)
        return state

    def get_random_k_state_lists(self, state_lists: np.ndarray, k: int):

        return state_lists[np.random.choice(len(state_lists), k)]

    def set_all_joints_state(self, object_id: int, joint_list: List[PbJointAxis], target_state_list: List[float]) -> None:
        assert len(joint_list) == len(target_state_list), "joint_list's length {} should be equal to target_state_list's {}".format(
            len(joint_list), len(target_state_list))
        for i in range(len(joint_list)):
            assert target_state_list[i] <= joint_list[i].jointUpperLimit and target_state_list[i] >= joint_list[i].jointLowerLimit, "index {}, target_state should be between [{}, {}], now {}".format(
                i, joint_list[i].jointLowerLimit, joint_list[i].jointUpperLimit, target_state_list[i])
            self.physicsClient.resetJointState(
                object_id, joint_list[i].jointIndex, target_state_list[i])
        return

    def get_joints_info(self, object_id) -> OrderedDict[int, Tuple[PbJoint, Tuple[int]]]:
        """Get the joints info from PyBullet

        Args:
            object_id:
                The return value of pybullet.loadURDF()
        Returns:
            joint_info: OrderedDict[jointIndex, Tuple[PbJoint, Tuple[subjointIndexes]]]

        """
        tree = Tree()
        tree.create_node('base', -1)
        joints_info = OrderedDict()
        num_joints = self.physicsClient.getNumJoints(object_id)
        mobile_roots = []
        for joint_idx in range(num_joints):
            v = self.physicsClient.getJointInfo(object_id, joint_idx)
            parent = v[-1]
            joint_name = v[1].decode('UTF-8')
            tree.create_node(joint_name, v[0], parent=parent, data=v)
            if v[2] in [0, 1]:
                mobile_roots.append(v[0])
        for node in mobile_roots:
            sub_nodes = tuple(tree.expand_tree(node))
            joints_info[node] = (
                PbJoint(*(tree.get_node(node).data), object_id), sub_nodes)
        return joints_info

    def _joint_local_to_global(self, joint_info: PbJoint) -> PbJointAxis:
        '''
        Get the global direction and postion of the joint from local coordinates.
        Denote World Frame as w, Parent Frame as p, Joint Frame as j.
        Then we have: JOINT_POSE_IN_W = W_to_J * JOINT_POSE_IN_J
        And W_to_J = W_to_P * P_to_J
        And P_to_J = Transform(JOINT_POSE_IN_P), W_to_P = Transform(PARENT_POSE_IN_W)
        '''
        if joint_info.parentIndex == -1:  # baselink
            parent_link_state = self.physicsClient.getBasePositionAndOrientation(
                joint_info.objectIndex)  # position, oritation
        else:
            parent_link_state = self.physicsClient.getLinkState(
                joint_info.objectIndex, joint_info.parentIndex)  # linkWorldPosition, linkWorldOrientation
        parent_link_trans: Transform = get_transform(
            parent_link_state[0], parent_link_state[1])  # World->Parant
        relative_trans: Transform = get_transform(
            joint_info.parentFramePos, joint_info.parentFrameOrn)  # Parent->Joint
        axis_trans = parent_link_trans * relative_trans  # World->Joint
        axis_global = axis_trans.rotation.as_matrix().dot(
            joint_info.jointAxis)  # R[World->Joint] * Joint_in_J
        axis_global /= np.sqrt(np.sum(axis_global ** 2))  # to unit vector
        point_on_axis = axis_trans.translation
        # moment = np.cross(point_on_axis, axis_global)

        return PbJointAxis(
            jointIndex=joint_info.jointIndex,
            jointName=joint_info.jointName,
            jointType=joint_info.jointType,
            jointLowerLimit=joint_info.jointLowerLimit,
            jointUpperLimit=joint_info.jointUpperLimit,
            jointState=-1,
            globalAxis=axis_global,
            globalPos=point_on_axis)

    def get_joints_info_global(self, object_id) -> List[PbJoint]:
        joints_info_global: List[PbJoint] = []
        joints_info: OrderedDict[int, Tuple[PbJoint,
                                            Tuple[int]]] = self.get_joints_info(object_id)
        for index, joint_info in joints_info.items():
            joint_axis_info_global = self._joint_local_to_global(joint_info[0])
            joint_info_global = joint_info[0]
            joint_info_global.globalAxis = joint_axis_info_global.globalAxis
            joint_info_global.globalPos = joint_axis_info_global.globalPos
            joint_info_global.jointState = self.physicsClient.getJointState(object_id, index)[
                0]
            joints_info_global.append(joint_info_global)

        return joints_info_global
    
    def get_links_info(self, object_id) -> List[PbLink]:
        link_state_list: List[PbLink] = []

        baselink_state_ = self.physicsClient.getBasePositionAndOrientation(object_id)
        baselink_state = PbLink(
                bodyUniqueId = object_id,
                linkIndex = -1,
                linkWorldPosition = baselink_state_[0],
                linkWorldOrientation = baselink_state_[1],
                localInertialFramePosition = np.zeros(3),
                localInertialFrameOrientation = np.zeros(4),
                worldLinkFramePosition = np.zeros(3),
                worldLinkFrameOrientation = np.zeros(4),
            )
        link_state_list.append(baselink_state)

        linkid_list = list(range(self.physicsClient.getNumJoints(object_id)))
        for linkid in linkid_list:
            link_state_ = self.physicsClient.getLinkState(object_id, linkid)
            link_state = PbLink(
                bodyUniqueId = object_id,
                linkIndex = linkid,
                linkWorldPosition = link_state_[0],
                linkWorldOrientation = link_state_[1],
                localInertialFramePosition = link_state_[2],
                localInertialFrameOrientation = link_state_[3],
                worldLinkFramePosition = link_state_[4],
                worldLinkFrameOrientation = link_state_[5],
            )
            link_state_list.append(link_state)

        return link_state_list

    def save_joints_info_as_json(self, joints_info: List[PbJoint], links_info: List[PbLink], save_path, prefix):
        joints_info_copy = copy.deepcopy(joints_info)
        links_info_copy = copy.deepcopy(links_info)
        json_info = {}
        with open(os.path.join(save_path, "{}_joints.json".format(prefix)), 'w') as f:
            for i in range(len(joints_info)):
                joints_info_copy[i].globalAxis = joints_info[i].globalAxis.tolist()
                joints_info_copy[i].globalPos = joints_info[i].globalPos.tolist()
                joints_info_copy[i].jointName = joints_info[i].jointName.decode(
                    'utf-8')
                joints_info_copy[i].linkName = joints_info[i].linkName.decode(
                    'utf-8')
                joints_info_copy[i] = asdict(joints_info_copy[i])
            for i in range(len(links_info)):
                links_info_copy[i].linkWorldPosition = links_info_copy[i].linkWorldPosition.tolist()
                links_info_copy[i].linkWorldOrientation = [_ for _ in links_info_copy[i].linkWorldOrientation]
                links_info_copy[i].localInertialFramePosition = [_ for _ in links_info_copy[i].localInertialFramePosition]
                links_info_copy[i].localInertialFrameOrientation = [_ for _ in links_info_copy[i].localInertialFrameOrientation]
                links_info_copy[i].worldLinkFramePosition = links_info_copy[i].worldLinkFramePosition.tolist()
                links_info_copy[i].worldLinkFrameOrientation = [_ for _ in links_info_copy[i].worldLinkFrameOrientation]
                links_info_copy[i] = asdict(links_info_copy[i])
            json_info["joints"] = joints_info_copy
            json_info["links"] = links_info_copy
            json.dump(json_info, f)

    # This is for adding some content to the already exist json
    def add_to_json(self, save_path, prefix, joints_info: List[PbJoint]):
        with open(os.path.join(save_path, "{}_joints.json".format(prefix)), 'r') as f:
            data = json.load(f)
        with open(os.path.join(save_path, "{}_joints_.json".format(prefix)), 'w') as f:
            for i in range(len(data)):
                joints_info[i].globalAxis = data[i]['globalAxis']
                joints_info[i].globalPos = data[i]['globalPos']
                joints_info[i].jointState = data[i]['jointState']
                joints_info[i].jointName = data[i]['jointName']
                joints_info[i].linkName = joints_info[i].linkName.decode(
                    'utf-8')
                joints_info[i] = asdict(joints_info[i])
            json.dump(joints_info, f)

    def camera_on_sphere(self, origin, radius, theta, phi) -> Transform:
        eye = np.r_[
            radius * sin(theta) * cos(phi),
            radius * sin(theta) * sin(phi),
            radius * cos(theta),
        ]
        target = np.array([0.0, 0.0, 0.0])
        # this breaks when looking straight down
        up = np.array([0.0, 0.0, 1.0])
        return Transform.look_at(eye, target, up) * origin.inverse()

    def get_dists(self, points1, points2):
        M, C = points1.shape
        N, _ = points2.shape
        dists = torch.sum(torch.pow(points1, 2), dim=-1).view(M, 1) + \
                torch.sum(torch.pow(points2, 2), dim=-1).view(1, N)
        dists -= 2 * torch.matmul(points1, points2.permute(1, 0))
        dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
        return torch.sqrt(dists).float()

    def farthest_point_sample(self, xyz, point_num):
        """
        Input:
            xyz: pointcloud data, [N, 3]
            point_num: number of samples
        Return:
            centroids: sampled pointcloud index, [point_num]
        """
        device = xyz.device
        ndataset, dimension = xyz.shape
        centroids = torch.zeros(point_num, dtype=torch.long).to(device)
        distance = torch.ones(ndataset, dtype=torch.float32).to(device) * 1e10
        farthest = torch.randint(
            0, ndataset, (1,), dtype=torch.long).to(device)
        for i in range(point_num):
            centroids[i] = farthest
            centroid = xyz[farthest, :].view(1, 3)
            dist = self.get_dists(xyz, centroid).squeeze()
            # dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def farthest_point_sample_batch(self, xyz, point_num):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            point_num: number of samples
        Return:
            centroids: sampled pointcloud index, [B, point_num]
        """
        device = xyz.device
        batch_size, ndataset, dimension = xyz.shape
        centroids = torch.zeros((batch_size, point_num), dtype=torch.long).to(device)
        distance = torch.ones((batch_size, ndataset), dtype=torch.float64).to(device) * 1e10
        farthest = torch.randint(
            0, ndataset, (batch_size, 1), dtype=torch.long).to(device)
        for i in range(point_num):
            centroids[:, i] = farthest.squeeze()
            # centroid = xyz[:, farthest, :].view(batch_size, 3)
            centroid = xyz[torch.arange(batch_size).unsqueeze(1), farthest, :].view(batch_size, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1].view(batch_size, 1)
        return centroids

    def gaussian_noise(self, point_cloud, noise_factor=1.0):
        """Adds Gaussian noise to the points in a point cloud.
        
        Args:
            point_cloud: A NumPy array of shape (N, 3) representing the point cloud,
            where N is the number of points.
            noise_factor: A scalar value that controls the amount of noise to add.
            A value of 1.0 adds noise with a standard deviation equal to the
            average distance between points in the point cloud. A value of 0.0
            adds no noise.
        
        Returns:
            A NumPy array of shape (N, 5) representing the point cloud with noise added.
        """
        # Compute the average distance between points in the point cloud
        distances = np.max(point_cloud[:,:3], axis=0) - np.min(point_cloud[:,:3], axis=0)
        mean_distance = distances / np.sqrt(point_cloud.shape[0])
        
        # Use the mean distance to compute the standard deviation of the noise
        std_dev = noise_factor * mean_distance
        
        # Add Gaussian noise to the point cloud
        noise = np.random.normal(0, std_dev, (point_cloud.shape[0], 3))
        return noise
    
    def add_occlusion(self, point_cloud, holes_num = 100, holes_size = 0.0001):

        # Set the length of the list
        length = point_cloud.shape[0]

        # Set the random factor that determines the length of each slice
        random_factor = holes_size

        # Create a list of True values
        lst = [True] * length

        # Set n slices of the list to False, with each slice's length being determined by the random factor
        n = holes_num
        for i in range(n):
            start = random.randint(0, length - 1)
            slice_length = int(length * random_factor)
            end = min(start + slice_length, length)
            lst[start:end] = [False] * (end - start)
        
        return point_cloud[lst]
    
    def add_outlier(self, point_cloud, outlier_num, outlier_range):

        if outlier_num<=0:
            return point_cloud

        points = []
        for i in range(outlier_num):
            outlier_points_num = np.random.randint(0, 1000)
            # Set the range of values for the x, y, and z coordinates
            x_range = (np.random.uniform(1) , np.random.uniform(0.01))
            y_range = (np.random.uniform(1) , np.random.uniform(0.01))
            z_range = (np.random.uniform(1) , np.random.uniform(0.01))
            x_offset = np.random.uniform(-outlier_range, outlier_range)
            y_offset = np.random.uniform(-outlier_range, outlier_range)
            z_offset = np.random.uniform(-outlier_range, outlier_range)
            # Generate the points
            points += [[random.gauss(*x_range) + x_offset, random.gauss(*y_range) + y_offset, random.gauss(*z_range) + z_offset, -1, -1, 0, 0, 0] for _ in range(outlier_points_num)]
        return np.concatenate((point_cloud, np.array(points)))

    def normalize(self, points):
        center = np.mean(points, axis=0)
        points -= center

        scale = np.mean(np.max(points, axis=0) - np.min(points, axis=0))
        points /= scale

        return points, center, scale
    
    def sample_or_pad(self, pc, point_num):
        num = pc.shape[0]
        if num == point_num:
            return pc
        elif num > point_num:
            return pc[random.sample(range(pc.shape[0]), point_num)]
        else:
            npc = np.zeros(shape=(point_num, pc.shape[1]))
            for i in range(0, num):
                npc[i] = pc[i]
            return npc