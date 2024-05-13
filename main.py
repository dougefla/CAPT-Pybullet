from contextlib import contextmanager
import sys
import time
import pybullet as pb
import pybullet_data
import os

import torch
from PbCamera import PbCamera
import multiprocessing as mp
import numpy as np

import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============ GENERAL SETTING ============
DATA_ROOT = r"/home/douge/Datasets/Motion_Dataset_v0/"
# DATA_ROOT = r"C:\Users\cvl\Desktop\fulian\Datasets\Motion_Dataset_v0"
INDEX = [8]  # 4, 8, 10, 11, 24, 27, 39, 41
'''
bucket          revolver	    kettle	        excavator           washing_machine	
motorbike	    fan             lamp	        laptop              swiss_army_knife	
bike            scissors	    cannon          screwdriver	        seesaw
closestool      water_bottle	wine_bottle	    folding_chair	    tank
handcart        door            swivel_chair    lighter             cabinet
plane           carton          eyeglasses      refrigerator	    watch
faucet          stapler         globe           car                 valve
skateboard      pen             swing           window              oven
rocking_chair   windmill        clock           helicopter	
'''
PROC_NUM = 5
SHOW_PLANE = False
RESUME = False
CUDA_ID = [0]

IMG_H = 1000
IMG_W = 1000

# ============== TASK SETTING ==============
# 'rgb', 'preprocessing', 'browse', 'test_set'
TASK = 'preprocessing'

# ============ BROWSING SETTING ============
BROWSE_TYPE = 'laptop'
BROWSE_INDEX = 1

# ========== PREPROCESSING SETTING ===========
# Debug mode. Set to `True` to draw pointcloud & set pybullet mode to `GUI`
PREP_OUTPUT_DIR = "preprocessed_test"
PNG_OUTPUT_DIR = "png"
NPY_OUTPUT_DIR = "npy"
PLY_OUTPUT_DIR = "ply"
JSON_OUTPUT_DIR = "json"
DEP_OUTPUT_DIR = "depth"
MSK_OUTPUT_DIR = "mask"
IS_DEBUG = False
RGB_ONLY = False
SAVE_PART = True
USE_NOISE = False
POINT_NUM = 2048
STATE_NUM = 10

# ============= RGB SETTING ==============
RGB_OUTPUT_DIR = "rgb"

# ============= TEST_SET SETTING ==============
TEST_OUTPUT_DIR = "test"
TEST_PNG_OUTPUT_DIR = "png"
TEST_NPY_OUTPUT_DIR = "npy"
TEST_PLY_OUTPUT_DIR = "ply"
TEST_JSON_OUTPUT_DIR = "json"
STATE_NUM = 10

@contextmanager
def suppress_stdout():

    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def browse(root_dir, object_type, index):

    # Connect to the Pybullet Server. Use pb.DIRECT/pb.GUI for non-graphical/graphical version
    connection_mode = pb.GUI

    # Create a camera with Pybullet Server connected
    pbcamera = PbCamera(connection_mode, pixelHeight=IMG_H, pixelWidth=IMG_W)

    # To use inner pybullet model, this is necessary
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Optional. Show plane.
    if SHOW_PLANE:
        planeId = pb.loadURDF("plane.urdf")

    # Fixed Inital Pose of the object.
    startPos = [0, 0, 1]
    startOrientation = pbcamera.physicsClient.getQuaternionFromEuler([0, 0, 0])

    np.random.seed(0)

    urdf_file = os.path.join(
        root_dir, 'urdf', object_type, "{:04d}".format(index), "syn.urdf")
    print("Loading urdf_file: {}".format(urdf_file))

    # Remove this line will raise "cannot extract anything useful from mesh " warning for some objects.
    # That's because they don't need non_motion parts.
    with suppress_stdout():
        object_id = pbcamera.physicsClient.loadURDF(
            urdf_file, startPos, startOrientation)
    joint_list = pbcamera.get_joints_info_global(object_id)
    state = pbcamera.generate_middle_state(joint_list)
    pbcamera.set_all_joints_state(object_id, joint_list, state)
    for i in range(10000):
        pbcamera.physicsClient.stepSimulation()
        time.sleep(1./240.)

    pbcamera.physicsClient.disconnect()


def task_func(urdf_per_proc, output_dir, root_dir, proc_id):

    # Connect to the Pybullet Server. Use pb.DIRECT/pb.GUI for non-graphical/graphical version
    connection_mode = pb.GUI if IS_DEBUG else pb.DIRECT

    # Create a camera with Pybullet Server connected
    pbcamera = PbCamera(pixelHeight=IMG_H, pixelWidth=IMG_W, connection_mode=connection_mode)

    # To use inner pybullet model, this is necessary
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    cuda_id = CUDA_ID if isinstance(CUDA_ID, int) else CUDA_ID[proc_id%len(CUDA_ID)]

    # Optional. Show plane.
    if SHOW_PLANE:
        planeId = pb.loadURDF("plane.urdf")

    if TASK == 'rgb':
        pbcamera.set_rpy(0, -45, 45)

    # Fixed Inital Pose of the object.
    startPos = [0, 0, 0]
    startOrientation = pbcamera.physicsClient.getQuaternionFromEuler([0, 0, 0])

    np.random.seed(0)

    # Process with each urdf file
    for task_id, urdf_dir in enumerate(urdf_per_proc):
        object_type = urdf_dir.split('/')[-2]
        object_type_dir = os.path.join(output_dir, object_type)
        if not os.path.isdir(object_type_dir):
            os.mkdir(object_type_dir)
        
        urdf_id = int(urdf_dir.split('/')[-1])

        # Load the urdf file
        urdf_file = os.path.join(
            root_dir, 'urdf', object_type, "{:04d}".format(urdf_id), "syn.urdf")

        if IS_DEBUG:
            print(urdf_file)

        # Remove this line will raise "cannot extract anything useful from mesh " warning for some objects.
        # That's because they don't need non_motion parts.
        with suppress_stdout():
            object_id = pbcamera.physicsClient.loadURDF(
                urdf_file, startPos, startOrientation)
        print("[{}] {} {}/{}".format(proc_id,
                urdf_dir, task_id, len(urdf_per_proc)))

        if TASK == 'preprocessing':
            # Operates the model.
            joint_list = pbcamera.get_joints_info_global(object_id)
            state_lists = pbcamera.generate_state_lists(joint_list, object_type, STATE_NUM)
            for state_id, state_list in enumerate(state_lists):
                prefix = "{}_{:04d}_{}".format(object_type, urdf_id, state_id)
                pbcamera.set_all_joints_state(
                    object_id, joint_list, state_list)

                pbcamera.set_rpy(0, -45, 45)
                img_rgb, img_depth, img_mask = pbcamera.get_image()
                if not os.path.isdir(os.path.join(object_type_dir, PNG_OUTPUT_DIR)):
                    os.mkdir(os.path.join(object_type_dir, PNG_OUTPUT_DIR))
                cv2.imwrite(os.path.join(
                    object_type_dir, PNG_OUTPUT_DIR, "{}.png".format(prefix)), img_rgb)

                if not RGB_ONLY:
                    # Resume, skip all files that already exist.
                    if RESUME and os.path.exists(os.path.join(os.path.join(object_type_dir, NPY_OUTPUT_DIR), "{}_pc.npy".format(prefix))):
                        continue

                    # Prepare the rpy camera pose list
                    rpy_list = pbcamera.uniform_camera_pose(
                        r_range=(0, 1, 1), p_range=(-180, 0, 30), y_range=(-180, 0, 30))

                    least_point_number = 10000

                    img_rgb_list = np.zeros((len(rpy_list), IMG_H, IMG_W, 4))
                    img_depth_list = np.zeros((len(rpy_list), IMG_H, IMG_W,))
                    img_mask_list = np.zeros((len(rpy_list), IMG_H, IMG_W))
                    points_list_for_full = []
                    points_sampled_list = np.zeros((len(rpy_list), least_point_number, 8))

                    # Camera rotate around the object, and capture and save all images
                    for rpy_id, rpy in enumerate(rpy_list):
                        try:
                            # Capture a photo with camera pose r,p,y
                            pbcamera.set_rpy(rpy[0], rpy[1], rpy[2])
                            pbcamera.physicsClient.stepSimulation()
                            img_rgb, img_depth, img_mask = pbcamera.get_image()
                            
                            # Create a point cloud using the images captured
                            # points [N, 8]
                            points = pbcamera.create_pointcloud(
                                img_depth, img_mask, img_rgb)
                            points_list_for_full.append(points)

                            # Store the captured images and point cloud
                            img_rgb_list[rpy_id, :,:,:] = img_rgb
                            img_depth_list[rpy_id, :,:] = img_depth
                            img_mask_list[rpy_id, :,:] = img_mask
                            if SAVE_PART:
                                points_sampled = pbcamera.sample_or_pad(points, least_point_number) # [N, 8]
                                points_sampled_list[rpy_id, :, :] = points_sampled
                                print("[{:1}] {:20} {:03d}/{:03d} {:02d}/{:02d} {:02d}/{:02d}".format(proc_id,
                                    object_type, task_id, len(urdf_per_proc), state_id, state_lists.shape[0], rpy_id, len(rpy_list)))
                            
                        except:
                            continue

                    # Process each frame if SAVE_PART is true
                    if SAVE_PART:
                        # Make a directory for part processing.
                        object_type_part_dir = os.path.join(object_type_dir, 'part')
                        if not os.path.isdir(object_type_part_dir):
                            os.makedirs(object_type_part_dir)
                        
                        # Using Batch FPS to sample the points
                        points_sampled_list_tensor = torch.from_numpy(
                            points_sampled_list).to("cuda:{}".format(cuda_id))
                        idx = pbcamera.farthest_point_sample_batch(
                            points_sampled_list_tensor[:, :, :3], point_num=POINT_NUM)
                        points_fps_sampled_list_tensor = points_sampled_list_tensor[torch.arange(len(rpy_list)).unsqueeze(1), idx, :]
                        
                        for rpy_id, rpy in enumerate(rpy_list):

                            points = points_fps_sampled_list_tensor[rpy_id].cpu().numpy()   #[N, 8]
                            # Skip the point cloud that is too incomplete, while keep the point cloud that is good enough.
                            
                            if points_list_for_full[rpy_id].shape[0] < least_point_number:
                                continue
                            # else:
                            #     least_point_number = points.shape[0]
                            points[:, :3], center, scale = pbcamera.normalize(
                                points[:, :3])

                            # Data Augmentation
                            if USE_NOISE:
                                points[:, :3] += pbcamera.gaussian_noise(
                                    points[:, :3], noise_factor=np.random.randint(1, 10))
                                points[:, :3] = pbcamera.add_occlusion(points, holes_num=np.random.randint(
                                    1, 1000), holes_size=np.random.random()*0.001)
                                points[:, :3] = pbcamera.add_outlier(
                                    points, outlier_num=np.random.randint(1, 50), outlier_range=2)

                            # Saving the results
                            part_prefix = prefix + \
                                "_part_{}".format(rpy_id)
                            if not os.path.isdir(os.path.join(object_type_part_dir, PNG_OUTPUT_DIR)):
                                os.mkdir(os.path.join(object_type_part_dir, PNG_OUTPUT_DIR))
                            cv2.imwrite(os.path.join(
                                object_type_part_dir, PNG_OUTPUT_DIR, "{}.png".format(part_prefix)), img_rgb_list[rpy_id])
                            
                            if not os.path.isdir(os.path.join(object_type_part_dir, DEP_OUTPUT_DIR)):
                                os.mkdir(os.path.join(object_type_part_dir, DEP_OUTPUT_DIR))
                            dep_save_name = os.path.join(
                                object_type_part_dir, DEP_OUTPUT_DIR, "{}.npy".format(part_prefix))
                            np.save(dep_save_name, img_depth_list[rpy_id])
                    
                            if not os.path.isdir(os.path.join(object_type_part_dir, MSK_OUTPUT_DIR)):
                                os.mkdir(os.path.join(object_type_part_dir, MSK_OUTPUT_DIR))
                            cv2.imwrite(os.path.join(
                                object_type_part_dir, MSK_OUTPUT_DIR, "{}.png".format(part_prefix)), img_mask_list[rpy_id])
                    
                            if not os.path.isdir(os.path.join(object_type_part_dir, NPY_OUTPUT_DIR)):
                                os.mkdir(os.path.join(
                                    object_type_part_dir, NPY_OUTPUT_DIR))
                            pbcamera.save_pointcloud_as_npy(
                                points, save_path=os.path.join(object_type_part_dir, NPY_OUTPUT_DIR), prefix=part_prefix)

                            if not os.path.isdir(os.path.join(object_type_part_dir, PLY_OUTPUT_DIR)):
                                os.mkdir(os.path.join(
                                    object_type_part_dir, PLY_OUTPUT_DIR))
                            pbcamera.save_pointcloud_as_ply(
                                points, save_path=os.path.join(object_type_part_dir, PLY_OUTPUT_DIR), prefix=part_prefix)

                            joints_info_global = pbcamera.get_joints_info_global(
                                object_id)
                            links_info_list = pbcamera.get_links_info(object_id)
                            # Normalization corespondingly with the normalization of point cloud
                            for joint_info_global in joints_info_global:
                                joint_info_global.normalize(
                                    center, scale)
                            for link_info in links_info_list:
                                link_info.normalize(center, scale)
                            if not os.path.isdir(os.path.join(object_type_part_dir, JSON_OUTPUT_DIR)):
                                os.mkdir(os.path.join(
                                    object_type_part_dir, JSON_OUTPUT_DIR))
                            pbcamera.save_joints_info_as_json(
                                joints_info_global, links_info_list, save_path=os.path.join(object_type_part_dir, JSON_OUTPUT_DIR), prefix=part_prefix)
                            if False and IS_DEBUG:
                                pbcamera.draw_pointcloud(
                                    points=points[:, :3], joints=joints_info_global)

                    # To create the complete point cloud
                    # Use GPU to accelerate FPS
                    pointcloud = torch.from_numpy(
                        np.concatenate(points_list_for_full)).to("cuda:{}".format(cuda_id))
                    idx = pbcamera.farthest_point_sample(
                        pointcloud[:, :3], point_num=POINT_NUM)
                    pointcloud = pointcloud[idx].cpu().numpy()

                    # Normalize the point cloud to scale (1,1,1) and center to (0,0,0)
                    pointcloud[:, :3], center, scale = pbcamera.normalize(
                        pointcloud[:, :3])

                    # Data Augmentation
                    if USE_NOISE:
                        pointcloud[:, :3] += pbcamera.gaussian_noise(
                            pointcloud[:, :3], noise_factor=np.random.randint(1, 10))
                        pointcloud = pbcamera.add_occlusion(pointcloud, holes_num=np.random.randint(
                            1, 1000), holes_size=np.random.random()*0.001)
                        pointcloud = pbcamera.add_outlier(
                            pointcloud, outlier_num=np.random.randint(1, 50), outlier_range=2)

                    # Save the whole point cloud.
                    if not os.path.isdir(os.path.join(object_type_dir, NPY_OUTPUT_DIR)):
                        os.mkdir(os.path.join(
                            object_type_dir, NPY_OUTPUT_DIR))
                    pbcamera.save_pointcloud_as_npy(
                        pointcloud, save_path=os.path.join(object_type_dir, NPY_OUTPUT_DIR), prefix=prefix)

                    if not os.path.isdir(os.path.join(object_type_dir, PLY_OUTPUT_DIR)):
                        os.mkdir(os.path.join(
                            object_type_dir, PLY_OUTPUT_DIR))
                    pbcamera.save_pointcloud_as_ply(
                        pointcloud, save_path=os.path.join(object_type_dir, PLY_OUTPUT_DIR), prefix=prefix)

                    joints_info_global = pbcamera.get_joints_info_global(
                        object_id)
                    links_info_list = pbcamera.get_links_info(object_id)
                    # Normalization corespondingly with the normalization of point cloud
                    for joint_info_global in joints_info_global:
                        joint_info_global.normalize(center, scale)
                    for link_info in links_info_list:
                        link_info.normalize(center, scale)

                    if not os.path.isdir(os.path.join(object_type_dir, JSON_OUTPUT_DIR)):
                        os.mkdir(os.path.join(
                            object_type_dir, JSON_OUTPUT_DIR))
                    pbcamera.save_joints_info_as_json(
                        joints_info_global, links_info_list, save_path=os.path.join(object_type_dir, JSON_OUTPUT_DIR), prefix=prefix)

                    if IS_DEBUG:
                        pbcamera.draw_pointcloud(
                            points=pointcloud[:, :3], joints=joints_info_global)
                
        elif TASK == 'test_set':
            # Operates the model.
            joint_list = pbcamera.get_joints_info_global(object_id)
            state_lists = pbcamera.generate_state_lists(joint_list, STATE_NUM)
            for state_id, state_list in enumerate(state_lists):
                prefix = "{}_{:04d}_{}".format(object_type, urdf_id, state_id)
                pbcamera.set_all_joints_state(
                    object_id, joint_list, state_list)

                pbcamera.set_rpy(0, -45, 45)
                img_rgb, img_depth, img_mask = pbcamera.get_image()
                if not os.path.isdir(os.path.join(object_type_dir, TEST_PNG_OUTPUT_DIR)):
                    os.mkdir(os.path.join(
                        object_type_dir, TEST_PNG_OUTPUT_DIR))
                cv2.imwrite(os.path.join(
                    object_type_dir, TEST_PNG_OUTPUT_DIR, "{}.png".format(prefix)), img_rgb)

                points = pbcamera.create_pointcloud(
                    img_depth, img_mask, img_rgb)

                if not os.path.isdir(os.path.join(object_type_dir, TEST_NPY_OUTPUT_DIR)):
                    os.mkdir(os.path.join(
                        object_type_dir, TEST_NPY_OUTPUT_DIR))
                pbcamera.save_pointcloud_as_npy(
                    points, save_path=os.path.join(object_type_dir, TEST_NPY_OUTPUT_DIR), prefix=prefix)

                if not os.path.isdir(os.path.join(object_type_dir, TEST_PLY_OUTPUT_DIR)):
                    os.mkdir(os.path.join(
                        object_type_dir, TEST_PLY_OUTPUT_DIR))
                pbcamera.save_pointcloud_as_ply(
                    points, save_path=os.path.join(object_type_dir, TEST_PLY_OUTPUT_DIR), prefix=prefix)

                joints_info_global = pbcamera.get_joints_info_global(
                    object_id)
                if not os.path.isdir(os.path.join(object_type_dir, TEST_JSON_OUTPUT_DIR)):
                    os.mkdir(os.path.join(
                        object_type_dir, TEST_JSON_OUTPUT_DIR))
                pbcamera.save_joints_info_as_json(
                    joints_info_global, save_path=os.path.join(object_type_dir, JSON_OUTPUT_DIR), prefix=prefix)
                if IS_DEBUG:
                    pbcamera.draw_pointcloud(
                        points=points[:, :3], joints=joints_info_global)

        # Remove the object to make room for the next object
        pbcamera.physicsClient.removeBody(object_id)
        pbcamera.physicsClient.resetSimulation()

    pbcamera.physicsClient.disconnect()

def get_task_list():
    root_dir = DATA_ROOT
    statics_file = os.path.join(root_dir, "statistics.txt")

    with open(statics_file, 'r') as f:
        lines = f.readlines()
        object_type_list = [object_type for object_type in (
            (lines[0]).split('\t'))[:-1]]
        object_type_num = [eval(object_num)
                           for object_num in ((lines[1]).split('\t'))[:-1]]
    
    def get_elements(mylist: list, index: list) -> list:
        return [mylist[i] for i in index]
    
    if not len(INDEX) == 0:
        object_type_list = get_elements(object_type_list, INDEX)
        object_type_num = get_elements(object_type_num, INDEX)
    
    urdf_list = []
    for object_type, object_num in zip(object_type_list, object_type_num):
        urdf_list += [os.path.join(root_dir, "urdf", object_type, "{:04d}".format(i)) for i in range(1, object_num+1)]

    # Split the task
    def split_integer(m, n):
        assert n > 0
        quotient = int(m / n)
        remainder = m % n
        if remainder > 0:
            return [quotient] * (n - remainder) + [quotient + 1] * remainder
        if remainder < 0:
            return [quotient - 1] * -remainder + [quotient] * (n + remainder)
        return [quotient] * n
    
    task_num_per_proc = split_integer(len(urdf_list), PROC_NUM)
    urdf_per_proc = []
    counter = 0
    for i in range(PROC_NUM):
        urdf_per_proc.append(
            urdf_list[counter:counter+task_num_per_proc[i]])
        counter += task_num_per_proc[i]
    
    return urdf_per_proc

def main():
    # Get dataset info. 2440 total samples for Motion_Dataset_v0.
    root_dir = DATA_ROOT
    statics_file = os.path.join(root_dir, "statistics.txt")

    # Set to browse mode. Will only show on Pybullet, no file write.
    if TASK == 'browse':
        browse(root_dir, BROWSE_TYPE, BROWSE_INDEX)
        return

    # with open(statics_file, 'r') as f:
    #     lines = f.readlines()
    #     object_type_list = [object_type for object_type in (
    #         (lines[0]).split('\t'))[:-1]]
    #     object_type_num = [eval(object_num)
    #                        for object_num in ((lines[1]).split('\t'))[:-1]]

    # def get_elements(mylist: list, index: list) -> list:
    #     return [mylist[i] for i in index]

    # if not len(INDEX) == 0:
    #     object_type_list = get_elements(object_type_list, INDEX)
    #     object_type_num = get_elements(object_type_num, INDEX)

    # Set the output dir
    if TASK == 'rgb':
        output_dir = os.path.join(DATA_ROOT, RGB_OUTPUT_DIR)
    elif TASK == 'preprocessing':
        output_dir = os.path.join(DATA_ROOT, PREP_OUTPUT_DIR)
    elif TASK == 'test_set':
        output_dir = os.path.join(DATA_ROOT, TEST_OUTPUT_DIR)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)



    # task_index_list = [i for i in range(len(object_type_list))]
    # task_num_per_proc = split_integer(len(object_type_list), PROC_NUM)
    # task_index_per_proc = []
    # counter = 0
    # for i in range(PROC_NUM):
    #     task_index_per_proc.append(
    #         task_index_list[counter:counter+task_num_per_proc[i]])
    #     counter += task_num_per_proc[i]

    urdf_per_proc = get_task_list()

    pool = mp.get_context("spawn").Pool(processes=PROC_NUM)
    # Run multiprocess
    for i in range(PROC_NUM):
        pool.apply_async(func=task_func, args=(
            urdf_per_proc[i], output_dir, root_dir, i))
    pool.close()
    pool.join()


if __name__ == '__main__':

    main()
