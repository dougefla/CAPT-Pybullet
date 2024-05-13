from typing import List
import open3d as o3d
from torch import Tensor
import numpy as np
import pickle

def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat
 
 
def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)
 
    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
 
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
 
    qTrans_Mat *= scale
    return qTrans_Mat
 
def get_arrow(center=[0,0,0],vec=[0,0,1],color = [0,1,0]):
    z_unit_Arr = np.array([0, 0, 1])
    begin = center
    end = np.add(center,vec)
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)
 
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])
 
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.03 * 1 ,
        cone_radius=0.01 * 1,
        cylinder_height=0.3 * 1,
        cylinder_radius=0.008 * 1
    )
    mesh_arrow.paint_uniform_color(color)
    mesh_arrow.compute_vertex_normals()
    
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))
    return mesh_arrow


def get_axis(pc, screw_axis, screw_moment):
    bound_max = pc.max(0)
    bound_min = pc.min(0)

    screw_point = np.cross(screw_axis, screw_moment)
    t_min = (bound_min - screw_point) / screw_axis
    t_max = (bound_max - screw_point) / screw_axis
    axis_index = np.argmin(np.abs(t_max - t_min))
    start_point = screw_point + screw_axis * t_min[axis_index]
    end_point = screw_point + screw_axis * t_max[axis_index]

    return start_point, end_point

def save_results(save_str:str, subset: str, points:np.ndarray, start_point:List[np.ndarray] = [], end_point:List[np.ndarray] = [], start_point_gt:List[np.ndarray] = [], end_point_gt:List[np.ndarray] = [], colors:List[np.ndarray] = []) -> None:
    result = {
        'subset': subset,
        'points': points,
        'start_point': start_point,
        'end_point': end_point,
        'start_point_gt': start_point_gt,
        'end_point_gt': end_point_gt,
        'colors': colors,
    }
    np.save(save_str, result)
    return

label_to_color = {
    0: [1,0,1],
    1: [0,0,1],
    2: [0,1,1],
    3: [0,1,0],
    4: [1,1,0],
    5: [1,0,0]
}

def seg_to_color_per_point(seg_p: np.ndarray) -> np.ndarray:
    C = len(seg_p)
    seg_c = np.argmax(seg_p)
    return label_to_color[seg_c%len(label_to_color)], seg_c

def seg_to_color(seg: np.ndarray) -> np.ndarray:
    if len(seg.shape) == 2:
        N, C = seg.shape
        color = np.zeros((N,3))
        seg_c_list = np.zeros((N))
        for i in range(N):
            color[i], seg_c_list[i] = seg_to_color_per_point(seg[i])
        return color
    else:
        N = seg.shape[0]
        color = np.zeros((N,3))
        for i in range(N):
            color[i] = label_to_color[seg[i]%len(label_to_color)]
        return color

def nocs_to_color(nocs: np.ndarray) -> np.ndarray:
    color = (np.clip((nocs+0.5), 0, 1))
    return color

def array_to_color(array: np.ndarray) -> np.ndarray:
    array = (array - array.min()) / (array.max() - array.min())
    if len(array.shape)==1:
        array = np.column_stack([array, np.zeros_like(array), np.zeros_like(array)])
    return np.clip(array, 0, 1)
    

def draw_pointcloud(points:np.ndarray, start_point:np.ndarray = [], end_point:np.ndarray = [], start_point_gt = [], end_point_gt = [], colors = []):
    geo_list = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if not len(colors)==0:
        colors = colors.astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geo_list.append(pcd)
    if not len(start_point)==0:
        if type(start_point[0])==np.float32:
            mesh_arrow = get_arrow(center=start_point, vec=end_point,color = [1,0,0])
            geo_list.append(mesh_arrow)
        else:
            for i in range(len(start_point)):
                mesh_arrow = get_arrow(center=start_point[i], vec=end_point[i],color = [1,0,0])
                geo_list.append(mesh_arrow)
    if not len(start_point_gt)==0:
        if type(start_point_gt[0])==np.float32:
            mesh_arrow_2 = get_arrow(center=start_point_gt, vec=end_point_gt, color = [0,1,0])
            geo_list.append(mesh_arrow_2)
        else:
            for i in range(len(start_point_gt)):
                mesh_arrow_2 = get_arrow(center=start_point_gt[i], vec=end_point_gt[i], color = [0,1,0])
                geo_list.append(mesh_arrow_2)
    o3d.visualization.draw_geometries(geo_list)

def draw_link(points:np.ndarray, center:np.ndarray, x_axis:np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray, colors = []):
    geo_list = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if not len(colors)==0:
        colors = colors.astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geo_list.append(pcd)

    x_arrow = get_arrow(center=center, vec=x_axis)
    y_arrow = get_arrow(center=center, vec=y_axis)
    z_arrow = get_arrow(center=center, vec=z_axis)
    geo_list.append(x_arrow)
    geo_list.append(y_arrow)
    geo_list.append(z_arrow)

    o3d.visualization.draw_geometries(geo_list)

def save_pointcloud(points:np.ndarray, start_point:np.ndarray = [], end_point:np.ndarray = [], save_dir = ''):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mesh_arrow = get_arrow(center=start_point, vec=end_point)
    o3d.io.write_triangle_mesh(filename = save_dir, mesh = mesh_arrow)

if __name__=='__main__':
    pointcloud = np.load("/home/douge/Datasets/Motion_Dataset_v0/pointcloud/windmill/windmill_0000_000080108_pc.npy")

