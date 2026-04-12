import argparse
import numpy as np
import random
import os.path as osp
import pickle
import os

parser = argparse.ArgumentParser(description='Dataset Preparation')
parser.add_argument('--root', type=str, default='/data/reiddatasets', help='path to data')
parser.add_argument('--dataset_name', type=str, default='ccv_s', help='vccr or ccvid or ccvs or ccvr')

args = parser.parse_args()

root = args.root
dataset_name = args.dataset_name



def prepare(root, dataset_name): 
    data_path = osp.join('/data/reiddatasets', dataset_name)
    if osp.exists(data_path):
        pass
    else:
        os.mkdir(data_path)
    if dataset_name == 'ccvid':
        
        root = osp.join(root, 'CCVID')
        prepare_ccvid(root=root)
    elif dataset_name == 'vccr':
        root = osp.join(root, 'VCCR')
        prepare_vccr(root=root)
    elif dataset_name == 'ccv_r':
        root = osp.join(root, 'RCCVReID')
        prepare_ccvr(root=root)
    elif dataset_name == 'ccv_s':
        root = osp.join(root, 'SCCVReID')
        prepare_ccvs(root=root)

def prepare_vccr(root):
    train_dir = osp.join(root, 'train')
    test_dir = osp.join(root, 'test_qg')

    # Training data
    ids = os.listdir(train_dir)
    train_set = []
    for id in ids:
        id_path = osp.join(train_dir, id)
        tracklets = os.listdir(id_path)
        for tracklet in tracklets:
            tracklet_path = osp.join(id_path, tracklet)
            imgs = os.listdir(tracklet_path)
            random_img = random.choice(imgs)
            cam_id = random_img.split('-')[1][1:]
            clothes_id = random_img.split('-')[2][1:]
            img_paths = []
            for img in imgs:
                img_path = osp.join(tracklet_path, img)
                img_paths.append(img_path)
            img_paths = sorted(img_paths, key=lambda s: s.split('-')[-1][1:-4])
            train_set.append(
                {'p_id': int(id), 
                 'img_paths': img_paths,
                 'cam_id': int(cam_id),
                 'clothes_id': int(clothes_id)
                 }
            )
    train_content = {
        'data': train_set,
        'num_pids': int(len(ids)),
    }
    with open(osp.join('/data/reiddatasets/vccr', 'train.pkl'), 'wb') as f:
            pickle.dump(train_content, f)

    # query and gallery data
    query_gallery= {'query': [], 'gallery': []}

    for key in list(query_gallery.keys()):
        data_set = []
        dir = osp.join(test_dir, key)
        ids = os.listdir(dir)
        for id in ids:
            id_path = osp.join(dir, id)
            cams = os.listdir(id_path)
            for cam in cams:
                cam_path = osp.join(id_path, cam)
                tracklets = os.listdir(cam_path)
                for tracklet in tracklets:
                    tracklet_path = osp.join(cam_path, tracklet)
                    imgs = os.listdir(tracklet_path)
                    random_img = random.choice(imgs)
                    cam_id = random_img.split('-')[1][1:]
                    clothes_id = random_img.split('-')[2][1:]
                    if len(imgs) < 8:
                        continue
                    img_paths = []
                    for img in imgs:
                        img_path = osp.join(tracklet_path, img)
                        img_paths.append(img_path)
                    img_paths = sorted(img_paths, key=lambda s: s.split('-')[-1][1:-4])
                    data_set.append(
                        {'p_id': int(id), 'img_paths': img_paths, 'cam_id': int(cam_id), 'clothes_id': int(clothes_id)}
                    )
        query_gallery[key] = {
            'data': data_set,
            'num_pids': int(len(ids)),
        }
        with open(osp.join('/data/reiddatasets/vccr', f'{key}.pkl'), 'wb') as f:
            pickle.dump(query_gallery[key], f)

def prepare_ccvid(root):
    modes = ['train', 'query', 'gallery']
    for mode in modes:
        data_path = osp.join(root, f'{mode}.txt')
        tracklets, _, num_pids, _, num_clothes, _, _ = process_ccvid(root, data_path, relabel=True)
        data_set = []
        for item in tracklets:
            data_set.append({
                'img_paths': item[0],
                'p_id': item[1],
                'cam_id': item[2],
                'clothes_id': item[3],
            })
        content = {
            'data': data_set,
            'num_clothes': num_clothes,
            'num_pids': num_pids
        }
        with open(osp.join('/data/reiddatasets/ccvid', f'{mode}.pkl'), 'wb') as f:
            pickle.dump(content, f)


def process_ccvid(root, data_path, relabel=False, clothes2label=None):
    tracklet_path_list = []
    pid_container = set()
    clothes_container = set()
    with open(data_path, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            tracklet_path, pid, clothes_label = new_line.split()
            tracklet_path_list.append((tracklet_path, pid, clothes_label))
            clothes = '{}_{}'.format(pid, clothes_label)
            pid_container.add(pid)
            clothes_container.add(clothes)
    pid_container = sorted(pid_container)
    clothes_container = sorted(clothes_container)
    pid2label = {pid:label for label, pid in enumerate(pid_container)}
    if clothes2label is None:
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

    num_tracklets = len(tracklet_path_list)
    num_pids = len(pid_container)
    num_clothes = len(clothes_container)

    tracklets = []
    num_imgs_per_tracklet = []
    pid2clothes = np.zeros((num_pids, len(clothes2label)))

    for tracklet_path, pid, clothes_label in tracklet_path_list:
        tracklet_path = osp.join(root, tracklet_path)
        img_paths = [osp.join(tracklet_path, img) for img in os.listdir(tracklet_path) if os.path.isfile(os.path.join(tracklet_path, img))]
        img_paths.sort()
        
        clothes = '{}_{}'.format(pid, clothes_label)
        clothes_id = clothes2label[clothes]
        pid2clothes[pid2label[pid], clothes_id] = 1
        if relabel:
            pid = pid2label[pid]
        else:
            pid = int(pid)
        session = tracklet_path.split('/')[0]
        cam = tracklet_path.split('_')[1]
        if session == 'session3':
            camid = int(cam) + 12
        else:
            camid = int(cam)

        num_imgs_per_tracklet.append(len(img_paths))
        tracklets.append((img_paths, pid, camid, clothes_id))

    num_tracklets = len(tracklets)

    return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label


def prepare_ccvr(root):
    all_clothes = set()
    train_txt = osp.join(root, 'withTrain', 'train.txt')
    if osp.exists(train_txt):
        with open(train_txt, 'r') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                _, pid, clothes_label = line.split()
                all_clothes.add(f"{pid}_{clothes_label}")
    for mode in ['query', 'gallery']:
        test_txt = osp.join(root, 'onlyTest', f'{mode}.txt')
        if osp.exists(test_txt):
            with open(test_txt, 'r') as f:
                for line in f:
                    line = line.rstrip()
                    if not line:
                        continue
                    _, pid, clothes_label = line.split()
                    all_clothes.add(f"{pid}_{clothes_label}")
    all_clothes = sorted(all_clothes)
    global_clothes2label = {clothes: label for label, clothes in enumerate(all_clothes)}
    num_total_clothes = len(global_clothes2label)

    if osp.exists(train_txt):
        tracklets_train, _, num_pids_train, _, _, pid2clothes_train, _ = process_ccvr(
            root, train_txt, relabel=True, clothes2label=global_clothes2label
        )
        train_data = []
        for item in tracklets_train:
            train_data.append({
                'img_paths': item[0],
                'mask_paths': item[4],
                'p_id': item[1],
                'cam_id': item[2],
                'clothes_id': item[3],
            })
        train_content = {
            'data': train_data,
            'num_clothes': num_total_clothes,
            'num_pids': num_pids_train,
            'pid2clothes': pid2clothes_train
        }
        train_save_path = osp.join('/data/reiddatasets/ccvr', 'train.pkl')
        os.makedirs(osp.dirname(train_save_path), exist_ok=True)
        with open(train_save_path, 'wb') as f:
            pickle.dump(train_content, f)
    else:
        pass
    for mode in ['query', 'gallery']:
        test_txt = osp.join(root, 'onlyTest', f'{mode}.txt')
        if not osp.exists(test_txt):
            continue
        
        tracklets_test, _, num_pids_test, _, _, pid2clothes_test, _ = process_ccvr(
            root, test_txt, relabel=True, clothes2label=global_clothes2label
        )
        
        test_data = []
        for item in tracklets_test:
            test_data.append({
                'img_paths': item[0],
                'mask_paths': item[4],
                'p_id': item[1],
                'cam_id': item[2],
                'clothes_id': item[3],
            })
        
        test_content = {
            'data': test_data,
            'num_clothes': num_total_clothes,
            'num_pids': num_pids_test,
            'pid2clothes': pid2clothes_test
        }
        
        test_save_path = osp.join('/data/reiddatasets/ccvr', f'{mode}.pkl')
        with open(test_save_path, 'wb') as f:
            pickle.dump(test_content, f)


def process_ccvr(root, data_path, relabel=False, clothes2label=None):
    tracklet_path_list = []
    pid_container = set()

    with open(data_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            video_path, original_pid, clothes_label = line.split()
            tracklet_path_list.append((video_path, original_pid, clothes_label))
            pid_container.add(original_pid)

    pid_container = sorted(pid_container)
    pid2label = {pid: label for label, pid in enumerate(pid_container)} if relabel else None
    num_pids = len(pid_container)
    num_clothes = len(clothes2label) if clothes2label else 0

    tracklets = []
    num_imgs_per_tracklet = []
    pid2clothes = np.zeros((num_pids, num_clothes))

    for video_path, original_pid, clothes_label in tracklet_path_list:
        path_parts = video_path.split('/')
        if len(path_parts) < 4:
            continue
        cam_id = path_parts[2]

        rgb_tracklet_path = osp.join(root, 'image', video_path)
        if not osp.exists(rgb_tracklet_path):
            continue
        img_names = [f for f in os.listdir(rgb_tracklet_path) if f.endswith('.jpg')]
        if not img_names:
            continue
        img_names.sort() 
        img_paths = [osp.join(rgb_tracklet_path, img_name) for img_name in img_names]
        mask_tracklet_path = osp.join(root, 'mask', video_path)
        mask_names = [f.replace('.jpg', '.png') for f in img_names]
        mask_paths = [osp.join(mask_tracklet_path, mask_name) for mask_name in mask_names]
        valid_pairs = []
        for img_p, mask_p in zip(img_paths, mask_paths):
            if osp.exists(img_p) and osp.exists(mask_p):
                valid_pairs.append((img_p, mask_p))
        if not valid_pairs:
            continue
        img_paths, mask_paths = zip(*valid_pairs)
        img_paths = list(img_paths)
        mask_paths = list(mask_paths)
        if len(img_paths) < 8:
            continue
        if relabel:
            pid = pid2label[original_pid]
        else:
            pid = int(original_pid)
        clothes_key = f"{original_pid}_{clothes_label}"
        if clothes_key not in clothes2label:
            continue
        clothes_id = clothes2label[clothes_key]
        pid_idx = pid2label[original_pid] if relabel else pid_container.index(original_pid)
        pid2clothes[pid_idx, clothes_id] = 1
        try:
            camid = int(cam_id)
        except ValueError:
            continue
        num_imgs_per_tracklet.append(len(img_paths))
        tracklets.append((img_paths, pid, camid, clothes_id, mask_paths))

    num_tracklets = len(tracklets)
    return (tracklets, num_tracklets, num_pids, num_imgs_per_tracklet,
            num_clothes, pid2clothes, clothes2label)


def prepare_ccvs(root):
    all_clothes = set()
    modes = ['train', 'query', 'gallery']
    
    for mode in modes:
        txt_path = osp.join(root, f'{mode}.txt')
        if not osp.exists(txt_path):
            continue
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                video_path, pid, clothes_label = line.split()
                all_clothes.add(f"{pid}_{clothes_label}")
    
    all_clothes = sorted(all_clothes)
    global_clothes2label = {clothes: label for label, clothes in enumerate(all_clothes)}
    num_total_clothes = len(global_clothes2label)

    for mode in modes:
        txt_path = osp.join(root, f'{mode}.txt')
        if not osp.exists(txt_path):
            continue
        
        tracklets, _, num_pids, _, _, pid2clothes, _ = process_ccvs(
            root, txt_path, relabel=True, clothes2label=global_clothes2label
        )
        
        data_set = []
        for item in tracklets:
            data_set.append({
                'img_paths': item[0],
                'mask_paths': item[4],
                'p_id': item[1],
                'cam_id': item[2],
                'clothes_id': item[3],
            })
        
        content = {
            'data': data_set,
            'num_clothes': num_total_clothes,
            'num_pids': num_pids,
            'pid2clothes': pid2clothes
        }
        
        save_path = osp.join('/data/reiddatasets/ccvs', f'{mode}.pkl')
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(content, f)

def process_ccvs(root, data_path, relabel=False, clothes2label=None):
    tracklet_path_list = []
    pid_container = set()

    with open(data_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            video_path, original_pid, clothes_label = line.split()
            video_path = video_path.replace('rgb/', '')
            tracklet_path_list.append((video_path, original_pid, clothes_label))
            pid_container.add(original_pid)
    
    pid_container = sorted(pid_container)
    pid2label = {pid: label for label, pid in enumerate(pid_container)} if relabel else None
    num_pids = len(pid_container)
    num_clothes = len(clothes2label) if clothes2label else 0

    tracklets = []
    num_imgs_per_tracklet = []
    pid2clothes = np.zeros((num_pids, num_clothes))

    for video_path, original_pid, clothes_label in tracklet_path_list:
        path_parts = video_path.split('/')
        if len(path_parts) < 3:
            continue
        cam_id = path_parts[-1]

        rgb_tracklet_path = osp.join(root, 'image', video_path)
        if not osp.exists(rgb_tracklet_path):
            continue
        img_names = [f for f in os.listdir(rgb_tracklet_path) if f.endswith('.jpg')]
        if not img_names:
            continue
        img_names.sort()
        img_paths = [osp.join(rgb_tracklet_path, img_name) for img_name in img_names]

        mask_tracklet_path = osp.join(root, 'mask', video_path)
        mask_names = [f.replace('.jpg', '.png') for f in img_names]
        mask_paths = [osp.join(mask_tracklet_path, mask_name) for mask_name in mask_names]

        valid_pairs = []
        for img_p, mask_p in zip(img_paths, mask_paths):
            if osp.exists(img_p) and osp.exists(mask_p):
                valid_pairs.append((img_p, mask_p))
        if not valid_pairs:
            continue
        img_paths, mask_paths = zip(*valid_pairs)
        img_paths = list(img_paths)
        mask_paths = list(mask_paths)
        if len(img_paths) < 8:
            continue
        if relabel:
            pid = pid2label[original_pid]
        else:
            pid = int(original_pid)

        clothes_key = f"{original_pid}_{clothes_label}"
        if clothes_key not in clothes2label:
            continue
        clothes_id = clothes2label[clothes_key]

        pid_idx = pid2label[original_pid] if relabel else pid_container.index(original_pid)
        pid2clothes[pid_idx, clothes_id] = 1

        try:
            camid = int(cam_id)
        except ValueError:
            continue
        
        num_imgs_per_tracklet.append(len(img_paths))
        tracklets.append((img_paths, pid, camid, clothes_id, mask_paths))

    num_tracklets = len(tracklets)

    return (tracklets, num_tracklets, num_pids, num_imgs_per_tracklet,
            num_clothes, pid2clothes, clothes2label)

if __name__ == "__main__":
    prepare(root, dataset_name)
    