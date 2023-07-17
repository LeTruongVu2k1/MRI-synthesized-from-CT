import SimpleITK as sitk
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import random
import cv2
import torch,pydicom


class Customized_CHAOS(Dataset):
    def __init__(self, path="CHAOS_preprocessed_v2", split='train', modals=('t2','ct'), transforms=None):
        super(Customized_CHAOS, self).__init__()
        for modal in modals:
            assert modal in {'t2','ct'}
        self.transforms = transforms
        self.split = split
        ####### Read T2 #######
        t2_img_path = sorted(glob.glob(f'{path}/{split}/t2/*/DICOM_anon/*.dcm*'))
        t2_label_path = sorted(glob.glob(f'{path}/{split}/t2/*/Ground/*.png*'))

        assert len(t2_img_path) == len(t2_label_path)
        assert list(map(img_to_label_t2, t2_img_path)) == t2_label_path

        self.t2_path = [[img, label] for img, label in zip(t2_img_path, t2_label_path)]

        ####### Read CT #######
        ct_img_path = sorted(glob.glob(f'{path}/{split}/ct/*/DICOM_anon/*.dcm*'))
        ct_label_path = sorted(glob.glob(f'{path}/{split}/ct/*/Ground/*.png*'))

        assert len(ct_img_path) == len(ct_label_path)
        self.ct_path = [[img, label] for img, label in zip(ct_img_path, ct_label_path)]

        self.new_perm()

        print("modal: {}, fold: {}, total size: t2: {}, ct: {}".format(modals,split,len(self.t2_path), len(self.ct_path)))

    def new_perm(self):
        x, y = len(self.t2_path), len(self.ct_path)
        self.randperm = torch.randperm(y)[:x]

    def __getitem__(self, idx):
        ######### Get T2 #########
        t2_img = self.t2_path[idx % len(self.t2_path)][0]

        t2_img = sitk.ReadImage(t2_img)
        t2_img = sitk.GetArrayFromImage(t2_img)[0]
        t2_img, t2_shape_mask = raw_preprocess(t2_img, True)

        t2_seg_mask = self.t2_path[idx % len(self.t2_path)][1]
        t2_seg_mask = sitk.ReadImage(t2_seg_mask)
        t2_seg_mask = label_preprocess(sitk.GetArrayFromImage(t2_seg_mask))

        ######### Get CT #########
        ct_img, ct_seg_mask = self.ct_path[self.randperm[idx]][0], self.ct_path[self.randperm[idx]][1]

        dcm = pydicom.dcmread(ct_img)
        wc = dcm.WindowCenter[0]
        ww = dcm.WindowWidth[0]
        slope = dcm.RescaleSlope
        intersept = dcm.RescaleIntercept
        low = wc - ww // 2
        high = wc + ww // 2
        img = dcm.pixel_array * slope + intersept
        img[img < low] = low
        img[img > high] = high
        img = (img - low) / (high - low)
        shape= img.copy()
        shape[shape != 0] = 1

        ct_img, ct_shape_mask = img, shape

        ct_seg_mask = sitk.ReadImage(ct_seg_mask)
        ct_seg_mask = sitk.GetArrayFromImage(ct_seg_mask)
        data = ct_seg_mask.astype(dtype=int)
        new_seg = np.zeros(data.shape, data.dtype)
        new_seg[data != 0] = 1
        ct_seg_mask = new_seg

        ######### Reset the randperm #########
        if idx == len(self) - 1:
            self.new_perm()

        ######### Some preprocessing steps #########
        if ct_img.shape[0] != 256:
            ct_img = cv2.resize(ct_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            ct_seg_mask = cv2.resize(ct_seg_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            ct_shape_mask = cv2.resize(ct_shape_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        if t2_img.shape[0] != 256:
            t2_img = cv2.resize(t2_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            t2_seg_mask = cv2.resize(t2_seg_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            t2_shape_mask = cv2.resize(t2_shape_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        t2_t_img = t2_img * t2_seg_mask
        ct_t_img = ct_img * ct_seg_mask

        if self.split == 'train':
            if random.random() > 0.5:
                t2_img, ct_img= cv2.flip(t2_img, 1), cv2.flip(ct_img, 1)
                t2_seg_mask, ct_seg_mask= cv2.flip(t2_seg_mask, 1), cv2.flip(ct_seg_mask, 1)
                t2_shape_mask, ct_shape_mask= cv2.flip(t2_shape_mask, 1), cv2.flip(ct_shape_mask, 1)
                t2_t_img, ct_t_img= cv2.flip(t2_t_img, 1), cv2.flip(ct_t_img, 1)

        #  scale to [-1,1]
        t2_img = (t2_img - 0.5) / 0.5
        t2_t_img = (t2_t_img - 0.5) / 0.5

        ct_img = (ct_img - 0.5) / 0.5
        ct_t_img = (ct_t_img - 0.5) / 0.5

        return (torch.from_numpy(t2_img).type(torch.FloatTensor).unsqueeze(dim=0),
                torch.from_numpy(t2_t_img).type(torch.FloatTensor).unsqueeze(dim=0),
                (torch.from_numpy(t2_shape_mask).type(torch.LongTensor).unsqueeze(dim=0),
                torch.from_numpy(t2_seg_mask).type(torch.LongTensor).unsqueeze(dim=0))), \
                \
               (torch.from_numpy(ct_img).type(torch.FloatTensor).unsqueeze(dim=0),
                torch.from_numpy(ct_t_img).type(torch.FloatTensor).unsqueeze(dim=0),
                (torch.from_numpy(ct_shape_mask).type(torch.LongTensor).unsqueeze(dim=0),
                torch.from_numpy(ct_seg_mask).type(torch.LongTensor).unsqueeze(dim=0)))

    def __len__(self):
        return min(len(self.t2_path), len(self.ct_path))
    

class CHAOS_inference(Dataset):
    def __init__(self, path="CHAOS_preprocessed_v2", modal='ct'):
        super(CHAOS_inference, self).__init__()
        assert modal == 'ct'
        
        
        ####### Read CT #######
        ct_img_path = sorted(glob.glob(f'{path}/**/DICOM_anon/*.dcm*', recursive=True))
        ct_label_path = sorted(glob.glob(f'{path}/**/Ground/*.png*', recursive=True))

        assert len(ct_img_path) == len(ct_label_path)
        assert len(ct_img_path) != 0
        
        self.ct_path = [[img, label] for img, label in zip(ct_img_path, ct_label_path)]
        
        print("modal: {}, total ct size: {}".format(modal, len(self.ct_path)))

    def __getitem__(self, idx):

        ######### Get CT #########
        ct_img, ct_seg_mask = self.ct_path[idx][0], self.ct_path[idx][1]
        ct_img_name, ct_seg_make_name = ct_img.split('/')[-1].split('.')[0], ct_seg_mask.split('/')[-1].split('.')[0]
        
        dcm = pydicom.dcmread(ct_img)
        wc = dcm.WindowCenter[0]
        ww = dcm.WindowWidth[0]
        slope = dcm.RescaleSlope
        intersept = dcm.RescaleIntercept
        low = wc - ww // 2
        high = wc + ww // 2
        img = dcm.pixel_array * slope + intersept
        img[img < low] = low
        img[img > high] = high
        img = (img - low) / (high - low)
        shape= img.copy()
        shape[shape != 0] = 1

        ct_img, ct_shape_mask = img, shape

        ct_seg_mask = sitk.ReadImage(ct_seg_mask)
        ct_seg_mask = sitk.GetArrayFromImage(ct_seg_mask)
        data = ct_seg_mask.astype(dtype=int)
        new_seg = np.zeros(data.shape, data.dtype)
        new_seg[data != 0] = 1
        ct_seg_mask = new_seg

        ######### Some preprocessing steps #########
        if ct_img.shape[0] != 256:
            ct_img = cv2.resize(ct_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            ct_seg_mask = cv2.resize(ct_seg_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            ct_shape_mask = cv2.resize(ct_shape_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        ct_t_img = ct_img * ct_seg_mask

        #  scale to [-1,1]
        ct_img = (ct_img - 0.5) / 0.5
        ct_t_img = (ct_t_img - 0.5) / 0.5

        return (torch.from_numpy(ct_img).type(torch.FloatTensor).unsqueeze(dim=0), 
                torch.from_numpy(ct_seg_mask).type(torch.LongTensor).unsqueeze(dim=0),
                ct_img_name, 
                ct_seg_make_name)

    def __len__(self):
        return len(self.ct_path)
    
    
def raw_preprocess(data, get_s=False):
    """
    :param data: [155,224,224]
    :return:
    """
    data = data.astype(dtype=float)
    data[data<50] = 0
    out = data.copy()
    out = (out - out.min()) / (out.max() - out.min())

    if get_s:
        share_mask = out.copy()
        share_mask[share_mask != 0] = 1
        return out, share_mask
    return out

def label_preprocess(data):
    data = data.astype(dtype=int)
    new_seg = np.zeros(data.shape, data.dtype)
    new_seg[(data > 55) & (data <= 70)] = 1
    return new_seg


def img_to_label_t2(path):
    return path.replace('DICOM_anon', 'Ground').replace('.dcm', '.png')