import os
import shutil

root = r'C:\Users\cvl\Desktop\fulian\Datasets\Motion_Dataset_v0\objects'
root_old = r'C:\Users\cvl\Desktop\fulian\Datasets\Motion_Dataset_v0\objects_'
# for category in os.listdir(root):
#     for id in os.listdir(os.path.join(root, category)):
#         files = os.listdir(os.path.join(root, category, id))
#         os.mkdir(os.path.join(root, category, id, 'part_objs'))
#         for file in files:
#             shutil.move(os.path.join(root, category, id, file), os.path.join(root, category, id, 'part_objs', file))
#             print(os.path.join(root, category, id, 'part_objs', file))

for category in os.listdir(root_old):
    for id in os.listdir(os.path.join(root_old, category)):
        files = os.listdir(os.path.join(root_old, category, id, 'part_objs'))
        for file in files:
            if '.mtl' in file:
                shutil.copy(os.path.join(root_old, category, id,  'part_objs', file), os.path.join(root, category, id, 'part_objs', file))
                print(os.path.join(root, category, id, 'part_objs', file))
