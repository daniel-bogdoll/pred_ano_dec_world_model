 
import torch
import numpy as np
import cv2
from collections import OrderedDict
import os
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import joblib
from PIL import Image
import matplotlib.pyplot as plt


def get_inputs(file_addr):
    file_format = file_addr.split('.')[-1]
    if file_format == 'mat':
        return sio.loadmat(file_addr, verify_compressed_data_integrity=False)['uv']
    elif file_format == 'npy':
        return np.load(file_addr)
    else:
        return cv2.imread(file_addr)


def img_tensor2numpy(img):
    # mutual transformation between ndarray-like imgs and Tensor-like images
    # both intensity and rgb images are represented by 3-dim data
    if isinstance(img, np.ndarray):
        return torch.from_numpy(np.transpose(img, [2, 0, 1]))
    else:
        return np.transpose(img, [1, 2, 0]).numpy()


def img_batch_tensor2numpy(img_batch):
    # both intensity and rgb image batch are represented by 4-dim data
    if isinstance(img_batch, np.ndarray):
        if len(img_batch.shape) == 4:
            return torch.from_numpy(np.transpose(img_batch, [0, 3, 1, 2]))
        else:
            return torch.from_numpy(np.transpose(img_batch, [0, 1, 4, 2, 3]))
    else:
        if len(img_batch.numpy().shape) == 4:
            return np.transpose(img_batch, [0, 2, 3, 1]).numpy()
        else:
            return np.transpose(img_batch, [0, 1, 3, 4, 2]).numpy()


class bbox_collate:
    def __init__(self, mode):
        self.mode = mode

    def collate(self, batch):
        if self.mode == 'train':
            return bbox_collate_train(batch)
        elif self.mode == 'test':
            return bbox_collate_test(batch)
        else:
            raise NotImplementedError


def bbox_collate_train(batch):
    batch_data = [x[0] for x in batch]
    batch_target = [x[1] for x in batch]
    return torch.cat(batch_data, dim=0), batch_target


def bbox_collate_test(batch):
    batch_data = [x[0] for x in batch]
    batch_target = [x[1] for x in batch]
    return batch_data, batch_target


def get_foreground(img, bboxes, patch_size):
    """
    Cropping the object area according to the bouding box, and resize to patch_size
    :param img: [#frame,c,h,w]
    :param bboxes: [#,4]
    :param patch_size: 32
    :return:
    """
    img_patches = list()
    if len(img.shape) == 3:
        for i in range(len(bboxes)):
            x_min, x_max = np.int(np.ceil(bboxes[i][0])), np.int(np.ceil(bboxes[i][2]))
            y_min, y_max = np.int(np.ceil(bboxes[i][1])), np.int(np.ceil(bboxes[i][3]))
            cur_patch = img[:, y_min:y_max, x_min:x_max]
            cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]), (patch_size, patch_size))
            img_patches.append(np.transpose(cur_patch, [2, 0, 1]))
            #print(f"Debug - Shape of cur_patch: {cur_patch.shape}")
        img_patches = np.array(img_patches)
       # print(f"Debug - Shape of img_patches: {img_patches.shape}")
    elif len(img.shape) == 4:
        for i in range(len(bboxes)):
            x_min, x_max = np.int(np.ceil(bboxes[i][0])), np.int(np.ceil(bboxes[i][2]))
            y_min, y_max = np.int(np.ceil(bboxes[i][1])), np.int(np.ceil(bboxes[i][3]))

            cur_patch_set = img[:, :, y_min:y_max, x_min:x_max]
            tmp_set = list()
            for j in range(img.shape[0]):  # temporal patches
                cur_patch = cur_patch_set[j]
                #print(f"Debug - Shape of cur_patch (frame {j}): {cur_patch.shape}")  # Add this debug statement
                cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]), (patch_size, patch_size))
                tmp_set.append(np.transpose(cur_patch, [2, 0, 1]))
               # print(f"Debug - Shape of resized cur_patch (frame {j}): {cur_patch.shape}")  # Add this debug statement

            cur_cube = np.array(tmp_set)  # spatial-temporal cube for each bbox
            img_patches.append(cur_cube)  # all spatial-temporal cubes in a single frame
            #print(f"Debug - Shape of cur_cube: {cur_cube.shape}")  # Add this debug statement

        img_patches = np.array(img_patches)
        #print(f"Debug - Shape of img_patches: {img_patches.shape}")
    return img_patches  # [num_bboxes,frames_num,C,patch_size, patch_size]



class common_dataset(Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, indice):
        raise NotImplementedError

    def _context_range(self, indice):
        """
        get a clip according to the indice (i.e., the frame to be predicted)
        :param indice: be consistent with __getitem__()
        :return: the frame indices in the clip
        """
        if self.border_mode == "predict":
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num
            end_idx = indice
            need_ctx_frames = self.context_frame_num + 1  # future frame prediction
        else:
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num

            if indice + self.context_frame_num > self.tot_frame_num - 1:
                end_idx = self.tot_frame_num - 1
            else:
                end_idx = indice + self.context_frame_num
            need_ctx_frames = 2 * self.context_frame_num + 1

        center_frame_video_idx = self.frame_video_idx[indice]
        clip_frames_video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        need_pad = need_ctx_frames - len(clip_frames_video_idx)

        if need_pad > 0:
            if start_idx == 0:
                clip_frames_video_idx = [clip_frames_video_idx[0]] * need_pad + clip_frames_video_idx
            else:
                clip_frames_video_idx = clip_frames_video_idx + [clip_frames_video_idx[-1]] * need_pad

        tmp = np.array(clip_frames_video_idx) - center_frame_video_idx
        offset = np.sum(tmp)

        if tmp[0] != 0 and tmp[-1] != 0:  # extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError

        if need_pad == 0 and offset == 0:
            idx = [x for x in range(start_idx, end_idx + 1)]
            return idx
        else:
            if self.border_mode == 'predict':
                if need_pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), need_pad) + idx
                return idx
            else:
                if need_pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)  # 把下一个视频的第一帧换成上一个视频的最后一帧
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if need_pad > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx, end_idx + 1)]
                        idx = [idx[0]] * need_pad + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx + 1)]
                        idx = idx + [idx[-1]] * need_pad
                        return idx


class ped_dataset(common_dataset):
    '''
    Loading dataset for UCSD ped2
    '''

    def __init__(self, dir, mode='train', context_frame_num=0, border_mode="hard",
                 file_format='.tif', all_bboxes=None, patch_size=32, of_dataset=False):
        super(ped_dataset, self).__init__()
        self.dir = dir
        self.mode = mode
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.frame_video_idx = list()
        self.tot_frame_num = 0
        self.context_frame_num = context_frame_num
        self.border_mode = border_mode
        self.file_format = file_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size

        self.of_dataset = of_dataset

        self.return_gt = False
        if mode == 'test':
            self.all_gt_addr = list()
            self.gts = OrderedDict()

        self._dataset_init()

    def __len__(self):
        return self.tot_frame_num

    def _dataset_init(self):
        if self.mode == 'train':
            data_dir = os.path.join(self.dir, 'training', 'frames') if not self.of_dataset \
                else os.path.join(self.dir, 'training', "flows")
        elif self.mode == 'test':
            data_dir = os.path.join(self.dir, 'testing', 'frames') if not self.of_dataset \
                else os.path.join(self.dir, 'testing', "flows")
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                if 'Train' in video_name:
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.file_format))
                    self.videos[video_name]['frame'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1
            

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

        elif self.mode == 'test':
            dir_list = glob.glob(os.path.join(data_dir, '*'))
            video_dir_list = []
            gt_dir_list = []
            for dir in sorted(dir_list):
                if '_gt' in dir:
                    gt_dir_list.append(dir)
                    self.return_gt = True
                else:
                    name = dir.split('/')[-1]
                    if 'Test' in name:
                        video_dir_list.append(dir)

            # load frames for test
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # load ground truth of frames
            if self.return_gt:
                for gt in sorted(gt_dir_list):
                    gt_name = gt.split('/')[-1]
                    self.gts[gt_name] = {}
                    self.gts[gt_name]['gt_frame'] = glob.glob(os.path.join(gt, '*.bmp'))
                    self.gts[gt_name]['gt_frame'].sort()

                # merge different frames of different videos into one list
                for _, cont in self.gts.items():
                    self.all_gt_addr += cont['gt_frame']

    def __getitem__(self, indice):
        if self.mode == "train":
            # frame indices in a clip
            frame_range = self._context_range(indice=indice)
            img_batch = []
            for idx in frame_range:
                # [h,w,c] -> [c,h,w] BGR
                cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                img_batch.append(cur_img)
            img_batch = np.array(img_batch)

            if self.all_bboxes is not None:
                # cropping
                img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
            img_batch = torch.from_numpy(img_batch)  # [num_bboxes,frames_num,C,patch_size, patch_size]

            return img_batch, torch.zeros(1)
        
        elif self.mode == "test":
            frame_range = self._context_range(indice=indice)
            img_batch = []
            for idx in frame_range:
                cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])  # [3,h,w]
                img_batch.append(cur_img)
            img_batch = np.array(img_batch)
            
            if self.all_bboxes is not None:
                img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
            
            img_batch = torch.from_numpy(img_batch)
            
            if self.return_gt:
                gt_batch = cv2.imread(self.all_gt_addr[indice], cv2.IMREAD_GRAYSCALE)
                gt_batch = torch.from_numpy(gt_batch)
                
            
            if self.return_gt:
                # img_batch [num_bboxes,frames_num,C,patch_size, patch_size]
                # gt_batch[h,w]
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)  # to unify the interface
        else:
            raise NotImplementedError


class CarlaDataset(common_dataset):
    def __init__(self, mode='train', context_frame_num=0, border_mode="hard", file_format='.png', all_bboxes=None, patch_size=32, of_dataset=False):
        super(CarlaDataset, self).__init__()
        self.mode = mode
        self.videos = OrderedDict()
        self.all_frame_addr = []
        self.frame_video_idx = []
        self.tot_frame_num = 0
        self.all_bboxes = all_bboxes
        self.patch_size =patch_size
        self.of_dataset = of_dataset
        self.context_frame_num = context_frame_num
        self.border_mode = border_mode
        self.file_format = file_format
        self.return_gt = False
        if mode == 'test':
            self.all_gt_addr = list()
            self.gts = OrderedDict()
        self._dataset_init()

    def __len__(self):
        return self.tot_frame_num

    def _dataset_init(self):
        if self.mode =='train':
            if self.of_dataset:
                self.data_dir = '/disk/vanishing_data/hb344/flows_carla'
                data_split = 'training' if self.mode == 'train' else 'testing'
                data_path = os.path.join(self.data_dir, data_split)
            else:
                 self.data_dir = '/disk/vanishing_data/hb344/Carla_Dataset_small/'
                 data_path = os.path.join(self.data_dir)
            town_dirs = glob.glob(os.path.join(data_path, 'Town*'))
            city_idx = 1
            image_count = 0  
            max_images = 20000000

            for town_dir in sorted(town_dirs):  
                
                town_name = os.path.basename(town_dir)
                scenario_dirs = glob.glob(os.path.join(town_dir, '*'))
                for scenario_dir in sorted(scenario_dirs):
                    scenario_name = os.path.basename(scenario_dir)
                    if scenario_name.isdigit():
                        image_path_pattern = os.path.join(scenario_dir, 'image', 'image*' + self.file_format)
                        frame_paths = sorted(glob.glob(image_path_pattern))
                        # Only add enough frames to reach the max_images limit
                        frame_paths = frame_paths[:max_images - image_count]
                        image_count += len(frame_paths)  # Update the counter
                        town_info = {
                            'path': image_path_pattern,
                            'frames': frame_paths,
                            'length': len(frame_paths)
                        }
                        self.videos[town_name + '/' + scenario_name] = town_info
                        self.frame_video_idx += [city_idx] * town_info['length']
                        city_idx += 1

            for town_info in self.videos.values():
                self.all_frame_addr.extend(town_info['frames'])
            self.tot_frame_num = len(self.all_frame_addr)
            
        if self.mode =='test':
            if self.of_dataset:
                data_dir = '/disk/vanishing_data/hb344/flows_carla/testing/testing/'
                gt_path = '/disk/vanishing_data/hb344/testing/'

                video_dir_list = sorted([dir for dir in glob.glob(os.path.join(data_dir, '00*'))])
                gt_dir_list = [os.path.join(gt_path, f'{i:04d}', 'ground-truth') for i in range(1, 27)]


                idx = 1
                for video_dir in video_dir_list:
                    video_name = os.path.basename(video_dir)
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video_dir
                    self.videos[video_name]['frame'] = sorted(glob.glob(os.path.join(video_dir, 'RGB_IMG', '*' + self.file_format)))
                    self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

                # merge different frames of different videos into one list
                for _, cont in self.videos.items():
                    self.all_frame_addr += cont['frame']
                self.tot_frame_num = len(self.all_frame_addr)
                if gt_dir_list:
                    self.return_gt = True
                    for gt_dir in gt_dir_list:
                        gt_name = os.path.basename(gt_dir).replace('ground-truth', 'GT')
                        self.gts[gt_name] = {}
                        self.gts[gt_name]['gt_frame'] = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
                    # Merge different frames of different videos into one list
                        for _, cont in self.gts.items():
                            self.all_gt_addr.extend(cont['gt_frame'])  # Use extend to merge lists

            if not self.of_dataset and self.mode =='test':
                self.data_dir = '/disk/vanishing_data/hb344/testing'
        
                data_path = os.path.join(self.data_dir)
                    
                video_dir_list = sorted([dir for dir in glob.glob(os.path.join(data_path, '00*'))])                
                gt_dir_list = [os.path.join(video_dir, 'ground-truth') for video_dir in video_dir_list]
                
                idx = 1
                for video_dir in video_dir_list:
                    video_name = os.path.basename(video_dir)
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video_dir
                    self.videos[video_name]['frame'] = sorted(glob.glob(os.path.join(video_dir, 'RGB_IMG', '*' + self.file_format)))
                    self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

                # merge different frames of different videos into one list
                for _, cont in self.videos.items():
                    self.all_frame_addr += cont['frame']
                self.tot_frame_num = len(self.all_frame_addr)

                if gt_dir_list:
                    self.return_gt = True
                    for gt_dir in gt_dir_list:
                        gt_name = os.path.basename(gt_dir).replace('ground-truth', 'GT')
                        self.gts[gt_name] = {}
                        self.gts[gt_name]['gt_frame'] = sorted(glob.glob(os.path.join(gt_dir, '*' + self.file_format)))

                    # Merge different frames of different videos into one list
                        for _, cont in self.gts.items():
                            self.all_gt_addr.extend(cont['gt_frame'])  # Use extend to merge lists

                    
            
    
    def __getitem__(self, indice):
        if self.mode == "train":
            # frame indices in a clip

            frame_range = self._context_range(indice=indice)
            img_batch = []
            for idx in frame_range:
                # [h,w,c] -> [c,h,w] BGR
                cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                img_batch.append(cur_img)
            img_batch = np.array(img_batch)

            if self.all_bboxes is not None:
                img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
            img_batch = torch.from_numpy(img_batch)  # [num_bboxes,frames_num,C,patch_size, patch_size]

            return img_batch, torch.zeros(1)

        elif self.mode == "test":
            frame_range = self._context_range(indice=indice)
            img_batch = []
            for idx in frame_range:
                cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])  # [3,h,w]
                img_batch.append(cur_img)
            img_batch = np.array(img_batch)
            
            if self.all_bboxes is not None:
                img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
            
            img_batch = torch.from_numpy(img_batch)
            
            if self.return_gt:
                gt_batch = cv2.imread(self.all_gt_addr[indice], cv2.IMREAD_GRAYSCALE)
                gt_batch = torch.from_numpy(gt_batch)
                
            
            if self.return_gt:
                # img_batch [num_bboxes,frames_num,C,patch_size, patch_size]
                # gt_batch[h,w]
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)  # to unify the interface
        else:
            raise NotImplementedError

    
    
class avenue_dataset(common_dataset):
    def __init__(self, dir, mode='train', context_frame_num=0, border_mode="hard",
                 file_format='.jpg',
                 all_bboxes=None, patch_size=32,
                 of_dataset=False):
        super(avenue_dataset, self).__init__()
        self.dir = dir
        self.mode = mode
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.frame_video_idx = list()
        self.tot_frame_num = 0
        self.context_frame_num = context_frame_num
        self.border_mode = border_mode
        self.file_format = file_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size

        self.of_dataset = of_dataset

        self.return_gt = False

        if mode == 'test':
            self.all_gt = list()
        self._dataset_init()

    def __len__(self):
        return self.tot_frame_num

    def _dataset_init(self):
        if self.mode == 'train':
            data_dir = os.path.join(self.dir, 'training', "frames") if not self.of_dataset \
                else os.path.join(self.dir, 'training', "flows")
        elif self.mode == 'test':
            data_dir = os.path.join(self.dir, 'testing', "frames") if not self.of_dataset \
                else os.path.join(self.dir, 'testing', "flows")

            gt_dir = os.path.join(self.dir, 'ground_truth_demo', 'testing_label_mask')
            if os.path.exists(gt_dir):
                self.return_gt = True
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

        elif self.mode == 'test':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # set address of ground truth of frames
            if self.return_gt:
                self.all_gt = [sio.loadmat(os.path.join(gt_dir, str(x + 1) + '_label.mat'))['volLabel'] for x in
                               range(len(self.videos))]
                self.all_gt = np.concatenate(self.all_gt, axis=1)
        else:
            raise NotImplementedError

    def __getitem__(self, indice):
        if self.mode == "train":
            frame_range = self._context_range(indice=indice)
            img_batch = []
            for idx in frame_range:
                # [h,w,c] -> [c,h,w] BGR
                cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                img_batch.append(cur_img)
            img_batch = np.array(img_batch)

            if self.all_bboxes is not None:
                # patch_size 32
                img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
            img_batch = torch.from_numpy(img_batch)  # [num_bboxes,frames_num,C,patch_size, patch_size]

            return img_batch, torch.zeros(1)

        elif self.mode == "test":
            frame_range = self._context_range(indice=indice)

            img_batch = []
            for idx in frame_range:
                cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])  # [3,h,w] BGR
                img_batch.append(cur_img)
            img_batch = np.array(img_batch)
            if self.all_bboxes is not None:
                img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
            img_batch = torch.from_numpy(img_batch)
            if self.return_gt:
                gt_batch = self.all_gt[0, indice]  # 最后一帧的gt mask
                gt_batch = torch.from_numpy(gt_batch)
            if self.return_gt:
                # img_batch [num_bboxes,frames_num,C,patch_size, patch_size],BGR
                # gt_batch[h,w]
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)  # to unify the interface
        else:
            raise NotImplementedError


class shanghaiTech_dataset(Dataset):
    def __init__(self, dir, mode='train', context_frame_num=0, border_mode='elastic', file_format='.jpg',
                 all_bboxes=None, patch_size=32, of_dataset=False):
        '''
        :param dir: The directory to load ShanghaiTech dataset
        mode: train/test dataset
        '''
        self.dir = dir
        self.mode = mode
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.frame_video_idx = list()
        self.tot_frame_num = 0
        self.context_frame_num = context_frame_num
        self.border_mode = border_mode
        self.file_format = file_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size

        self.of_dataset = of_dataset

        self.return_gt = False
        self.save_scene_idx = list()
        self.scene_idx = list()
        self.scene_num = 0
        if mode == 'test':
            self.all_gt = list()
        self.dataset_init()
        pass

    def __len__(self):
        return self.tot_frame_num

    def dataset_init(self):
        if self.mode == 'train':
            data_dir = os.path.join(self.dir, 'training', 'frames') if not self.of_dataset \
                else os.path.join(self.dir, 'training', 'flows')
        elif self.mode == 'test':
            data_dir = os.path.join(self.dir, 'testing', 'frames') if not self.of_dataset \
                else os.path.join(self.dir, 'testing', 'flows')
            gt_dir = os.path.join(self.dir, 'ground_truth_demo', 'test_frame_mask')
            if os.path.exists(gt_dir):
                self.return_gt = True
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1  # 视频idx
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1
                self.save_scene_idx += [int(video_name[:2])] * len(
                    self.videos[video_name]['frame'])  # scene idx
                self.scene_idx += [1] * len(self.videos[video_name]['frame'])  # frames are processed by scene idx

            self.scene_num = len(set(self.scene_idx))
            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

        elif self.mode == 'test':
            idx = 1  # video idx
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))

            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1
                # scene idx
                self.save_scene_idx += [int(video_name[:2])] * len(self.videos[video_name]['frame'])
                # frames are processed by scene idx
                self.scene_idx += [1] * len(self.videos[video_name]['frame'])

            self.scene_num = len(set(self.scene_idx))
            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # load ground truth of frames
            if self.return_gt:
                gt_dir_list = glob.glob(os.path.join(gt_dir, '*'))
                for gt in sorted(gt_dir_list):
                    self.all_gt.append(np.load(gt))

                # merge different frames of different videos into one list, only support frame gt now due to memory issue
                self.all_gt = np.concatenate(self.all_gt, axis=0)
        else:
            raise NotImplementedError

    def _context_range(self, indice):
        if self.border_mode == 'predict':
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num
            end_idx = indice
            need_context_num = self.context_frame_num + 1
        else:
            # check head and tail
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num

            if indice + self.context_frame_num > self.tot_frame_num - 1:
                end_idx = self.tot_frame_num - 1
            else:
                end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1

        center_idx = self.frame_video_idx[indice]
        video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        pad = need_context_num - len(video_idx)
        if pad > 0:
            if start_idx == 0:
                video_idx = [video_idx[0]] * pad + video_idx
            else:
                video_idx = video_idx + [video_idx[-1]] * pad
        tmp = np.array(video_idx) - center_idx
        offset = tmp.sum()
        if tmp[0] != 0 and tmp[-1] != 0:  # extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError
        if pad == 0 and offset == 0:  # all frames are from the same video
            idx = [x for x in range(start_idx, end_idx + 1)]
            return idx
        else:
            if self.border_mode == 'predict':
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), pad) + idx
                return idx
            else:
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if pad > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx - offset, end_idx + 1)]
                        idx = [idx[0]] * pad + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx - offset + 1)]
                        idx = idx + [idx[-1]] * pad
                        return idx

    def __getitem__(self, indice):
        if self.mode == 'train':
            frame_range = self._context_range(indice=indice)
            img_batch = []
            for idx in frame_range:
                cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                img_batch.append(cur_img)
            img_batch = np.array(img_batch)
            if self.all_bboxes is not None:
                img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice],
                                           patch_size=self.patch_size)
            img_batch = torch.from_numpy(img_batch)

            return img_batch, torch.zeros(1)  # to unify the interface
        
    
        elif self.mode == 'test':
            frame_range = self._context_range(indice=indice)
            img_batch = []
            for idx in frame_range:
                cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                img_batch.append(cur_img)
            img_batch = np.array(img_batch)
            if self.all_bboxes is not None:
                img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice],
                                           patch_size=self.patch_size)
            img_batch = torch.from_numpy(img_batch)
            if self.return_gt:
                gt_batch = np.array([self.all_gt[indice]])
                gt_batch = torch.from_numpy(gt_batch)

            if self.return_gt:
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)  # to unify the interface
        else:
            raise NotImplementedError
        



def get_dataset(dataset_name, dir, mode='train', context_frame_num=0, border_mode='hard',
                all_bboxes=None, patch_size=32, of_dataset=False):
    if not of_dataset:
        img_ext = {"ped2": ".tif", "avenue": ".jpg", "shanghaitech": ".jpg", "carla_local" : ".png"}[dataset_name]
    else:
        img_ext = ".npy"

    if dataset_name == "ped2":
        dataset = ped_dataset(dir=dir, context_frame_num=context_frame_num, mode=mode, border_mode=border_mode,
                              all_bboxes=all_bboxes, patch_size=patch_size, file_format=img_ext,
                              of_dataset=of_dataset)
    elif dataset_name == 'avenue':
        dataset = avenue_dataset(dir=dir, context_frame_num=context_frame_num, mode=mode, border_mode=border_mode,
                                 all_bboxes=all_bboxes, patch_size=patch_size, file_format=img_ext,
                                 of_dataset=of_dataset)
    elif dataset_name == 'shanghaitech':
        dataset = shanghaiTech_dataset(dir=dir, context_frame_num=context_frame_num, mode=mode, border_mode=border_mode,
                                       all_bboxes=all_bboxes, patch_size=patch_size, file_format=img_ext,
                                       of_dataset=of_dataset)
    elif dataset_name == 'carla_local':
        dataset = CarlaDataset(
                    mode=mode,
                    context_frame_num=context_frame_num,
                    border_mode=border_mode,
                    all_bboxes=all_bboxes,
                    patch_size=patch_size,
                    file_format=img_ext,
                    of_dataset = of_dataset)
    else:
        raise NotImplementedError

    return dataset


transform = transforms.Compose([
    transforms.ToTensor(),
])


class Chunked_sample_dataset(Dataset):
    def __init__(self, chunk_file, last_flow=False, transform=transform):
        super(Chunked_sample_dataset, self).__init__()
        self.chunk_file = chunk_file
        self.last_flow = last_flow

        # dict(sample_id=[], appearance=[], motion=[], bbox=[], pred_frame=[])
        self.chunked_samples = joblib.load(self.chunk_file)

        self.chunked_samples_appearance = self.chunked_samples["appearance"]
        self.chunked_samples_motion = self.chunked_samples["motion"]
        self.chunked_samples_bbox = self.chunked_samples["bbox"]
        self.chunked_samples_pred_frame = self.chunked_samples["pred_frame"]
        self.chunked_samples_id = self.chunked_samples["sample_id"]

        self.transform = transform

    def __len__(self):
        return len(self.chunked_samples_id)

    def __getitem__(self, indice):
        appearance = self.chunked_samples_appearance[indice]
        motion = self.chunked_samples_motion[indice]
        bbox = self.chunked_samples_bbox[indice]
        pred_frame = self.chunked_samples_pred_frame[indice]

        # [#frame,h,w,c] to [h,w,#frame,c]
        x = np.transpose(appearance, [1, 2, 0, 3])
        x = np.reshape(x, (x.shape[0], x.shape[1], -1))

        y = motion[1:] if not self.last_flow else motion[-1:]
        y = np.transpose(y, [1, 2, 0, 3])
        y = np.reshape(y, (y.shape[0], y.shape[1], -1))

        return self.transform(x), self.transform(y), \
               bbox.astype(np.float32), pred_frame, indice
