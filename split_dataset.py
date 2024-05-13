import os
import random
'''
The file structure is:
- root
    - bike
        - json
            - bike_0000_0_joints.json
            - ...
        - npy
            - bike_0000_0_pc.npy
            - ...
        - ply
            - bike_0000_0_pc.ply
            - ...
        - png
            - bike_0000_0.png
        - parts
            - json
                - bike_0000_0_part_0_joints.json
                - ...
            - npy
                - bike_0000_0_part_0_pc.npy
                - ...
            - ply
                - bike_0000_0_part_0_pc.ply
                - ...
            - png
                - bike_0000_0_part_0.png
                - ...
    - bucket
    - ...
'''

skip_model_list = {}
skip_model_list['oven'] = ['0014']
skip_model_list['washing_machine'] = ['0057']
skip_model_list['eyeglasses'] = []

def divide_items(items, a, b, c):
    random.seed(0)
    total = len(items)
    counts = [int(total * a), int(total * b), int(total * c)]
    counts[0] += total - sum(counts)  # adjust for rounding errors
    random.shuffle(items)
    return [items[:counts[0]], items[counts[0]:counts[0]+counts[1]], items[counts[0]+counts[1]:]]

def split_dataset(root, subset, ratio, sample_rate, is_part=True):
    random.seed(0)
    model_predix_dict = {}
    subset_folder = os.path.join(root, subset, 'part') if is_part else os.path.join(root, subset)
    train_ratio, val_ratio, test_ratio = ratio

    subset_npy_folder = os.path.join(subset_folder, 'npy')
    subset_json_folder = os.path.join(subset_folder, 'json')
    for item in os.listdir(subset_npy_folder):
        prefix = item[:-7]
        model_id = prefix.split('_')[-4] if is_part else prefix.split('_')[-2]
        if os.path.exists(os.path.join(subset_json_folder, prefix+'_joints.json')):
            if not model_id in skip_model_list[subset]:
                if not model_id in model_predix_dict.keys():
                    model_predix_dict[model_id] = []
                    model_predix_dict[model_id].append(prefix)
                else:
                    model_predix_dict[model_id].append(prefix)

    train_model_ids, val_model_ids, test_model_ids = divide_items(list(model_predix_dict.keys()), train_ratio, val_ratio, test_ratio)
    train_prefix_list = []
    for i in train_model_ids:
        train_prefix_list+=random.sample(model_predix_dict[i], int(sample_rate[0]*len(model_predix_dict[i])))
    val_prefix_list = []
    for i in val_model_ids:
        val_prefix_list+=random.sample(model_predix_dict[i], int(sample_rate[1]*len(model_predix_dict[i])))
    test_prefix_list = []
    for i in test_model_ids:
        test_prefix_list+=random.sample(model_predix_dict[i], int(sample_rate[2]*len(model_predix_dict[i])))

    def save_split(save_path, prefix_list, max_num, name):
        with open(os.path.join(save_path, name), 'w') as f:
            random.shuffle(prefix_list)
            f.writelines([item+'\n' for item in prefix_list[:max_num]])

    save_split(subset_folder, train_prefix_list, -1, 'train.txt')
    save_split(subset_folder, val_prefix_list, -1, 'val.txt')
    save_split(subset_folder, test_prefix_list, -1, 'test.txt')

if __name__ == '__main__':
    # data_root = r"C:\Users\cvl\Desktop\fulian\Datasets\Motion_Dataset_v0\preprocessed_2048"
    data_root = "/home/douge/Datasets/Motion_Dataset_v0/preprocessed_2048"

    split_dataset(data_root, subset='eyeglasses', ratio=(0.8, 0.1, 0.1), sample_rate=(1, 1, 1), is_part=True)