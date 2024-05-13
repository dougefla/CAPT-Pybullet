def make_double_sided(file_path, new_file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        if line.startswith('f '):
            # split the face line into individual vertex indices
            face = line.split()[1:]
            if len(face) == 3:
                # if it's a triangle face, duplicate it and flip the vertex order
                new_face = face[::-1]
                new_lines.append(f'f {" ".join(face)}\n')
                new_lines.append(f'f {" ".join(new_face)}\n')
            elif len(face) == 4:
                # if it's a quad face, split it into two triangles
                tri1 = face[:3]
                tri2 = face[1:]
                tri2.append(face[0])
                new_lines.append(f'f {" ".join(tri1)}\n')
                new_lines.append(f'f {" ".join(tri2)}\n')
            else:
                # unsupported face format
                raise ValueError(f'Unsupported face format: {line}')
        else:
            # non-face line, copy it over as is
            new_lines.append(line)

    # write the modified file to disk
    with open(new_file_path, 'w') as f:
        print(new_file_path)
        f.writelines(new_lines)

import os
import shutil

# Define the root directory of the original OBJ files
root_dir = r'/home/douge/Datasets/Motion_Dataset_v0/objects'

# Define the root directory where the modified OBJ files will be saved
new_root_dir = r'/home/douge/Datasets/Motion_Dataset_v0/new_objects'


# Define the function that makes the faces double-sided
# Same function as in the previous answer

# Traverse the directory structure and modify the OBJ files
for category in os.listdir(root_dir):
    category_dir = os.path.join(root_dir, category)
    if not os.path.isdir(category_dir):
        continue
    new_category_dir = os.path.join(new_root_dir, category)
    os.makedirs(new_category_dir, exist_ok=True)
    for obj_id in os.listdir(category_dir):
        obj_dir = os.path.join(category_dir, obj_id, 'part_objs')
        if not os.path.isdir(obj_dir):
            continue
        new_obj_dir = os.path.join(new_category_dir, obj_id, 'part_objs')
        os.makedirs(new_obj_dir, exist_ok=True)
        for part_obj in os.listdir(obj_dir):
            if not part_obj.endswith('.obj'):
                continue
            part_obj_path = os.path.join(obj_dir, part_obj)
            new_part_obj_path = os.path.join(new_obj_dir, part_obj)
            make_double_sided(part_obj_path, new_part_obj_path)