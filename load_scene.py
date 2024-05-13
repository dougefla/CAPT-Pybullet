import numpy as np
from PbCamera import PbCamera

root = "/home/fla/datasets/Motion_Dataset_v0/scenes/1a82d7bdafd344d18eb183efc8e2ca0e.npz"
a = np.load(root)
pb = PbCamera()
pb.draw_pointcloud(points=a["pc_end"])
print(1)