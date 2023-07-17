from data_loader import CHAOS_inference
from models import Generator
from torch.utils.data import DataLoader
import nibabel as nib
from tqdm import tqdm
import argparse
import os
import glob
import numpy as np
import torch
import shutil
import gdown
import zipfile

# Downstream task - nnUNet segmentation
def nnUNet_predict(input_name, output_name, device):
    os.system(f"nnUNetv2_predict -d Dataset001_Liver -i '{input_name}' -o '{output_name}' -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans -device {device}")

def load_gen(checkpoint, device):
    genCT2MR_loaded = Generator(1, 64, 2, 3, True, True).to(device)
    genCT2MR_loaded.load_state_dict(checkpoint['genCT2MR_state_dict'])
    return genCT2MR_loaded

# convert images into nib files for segmenting
def store_prediction(ct_img, ct_seg_mask, ct_img_name, ct_seg_mask_name, t2_fake, ct_npy_path, prediction_path, idx):
    affine = np.eye(4)
    for i in range(len(ct_img)):
        # Storing .npy ct image
        np.save(f'{ct_npy_path}/{ct_img_name[i]}.npy', ct_img[i][0])
        
        # Storing mask 
        np.save(f'{ct_npy_path}/{ct_seg_mask_name[i]}.npy', ct_seg_mask[i][0])
        
        # Storing predictions
        # t2_fake[i] = (t2_fake[i]).squeeze()
        t2 = t2_fake[i]
        t2 = np.expand_dims(t2.squeeze(), 2)
        nii_file = nib.Nifti1Image(t2, affine)
        nib.save(nii_file, f'{prediction_path}/{ct_img_name[i]}_0000.nii.gz')
        
# def store_prediction(ct_img, ct_seg_mask, t2_fake, ct_npy_path, prediction_path, idx):
#     affine = np.eye(4)
#     for i, (ct, seg_mask, t2) in enumerate(zip(ct_img, ct_seg_mask, t2_fake)):
#         # Storing .npy ct image
#         np.save(f'{ct_npy_path}/{i+idx}.npy', ct[0])
        
#         # Storing mask 
#         np.save(f'{ct_npy_path}/Ground_{i+idx}.npy', seg_mask[0])
        
#         # Storing predictions
#         t2 = np.expand_dims(t2.squeeze(), 2)
#         nii_file = nib.Nifti1Image(t2, affine)
#         nib.save(nii_file, f'{prediction_path}/{i+idx}_0000.nii.gz')
        
def inference_and_segmentation(args):
    # Downloading checkpoint
    if args.checkpoint.split('/')[-1] == '19.6_saved_models_24576.pt':
        url = 'https://drive.google.com/uc?export=download&id=1Z67wViOh3khnEvkaxaUdsd_Uup_VJadL'
        output = '19.6_saved_models_24576.pt'
        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile('19.6_saved_models_24576.pt', 'r') as zip_ref:
            checkpoints_dir = '/'.join(args.checkpoint.split('/')[:-1])
            zip_ref.extractall(checkpoints_dir)
    
    # Loading dataset
    dataset = CHAOS_inference(path=args.path, modal='ct')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Loading Generator
    checkpoint = args.checkpoint
    checkpoint = torch.load(checkpoint, map_location='cpu')
    gen = load_gen(checkpoint, args.device)

    # Remove already exists dir
    parent_dir = args.save_dir
    if os.path.exists(parent_dir):
        shutil.rmtree(parent_dir)
        
    # MR-synthesizing and segmenting
    prediction_path = os.path.join(parent_dir, 'synthesized_MRI')
    segmentation_path = os.path.join(parent_dir, 'segmentations')
    ct_npy_path = os.path.join(parent_dir, 'ct_npy_path')

    os.makedirs(prediction_path)
    os.makedirs(segmentation_path)
    os.makedirs(ct_npy_path)

    # Setting nnUNet environment variable for segmenting
    os.environ['nnUNet_results'] = f'{args.nnUNet_dir}/nnUNet_results'

    # Inferencing 
    idx = 0
    for (ct_img, ct_seg_mask, ct_img_name, ct_seg_mask_name) in dataloader:
        t2_fake = gen(ct_img.to(args.device), mode='test').detach().cpu().numpy().copy()
        
        store_prediction(ct_img, ct_seg_mask, ct_img_name, ct_seg_mask_name, t2_fake, ct_npy_path, prediction_path, idx)
        idx += len(t2_fake)

        # Segmenting using nnUNet
        nnUNet_predict(prediction_path, segmentation_path, args.device)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-path', type=str, default='MRI-synthesized-from-CT/ct', help="Folder contains data for inferencing")
    parser.add_argument('-checkpoint', type=str, default='MRI-synthesized-from-CT/checkpoints/19.6_saved_models_24576.pt', help="Pretrained model for inferencing")
    # parser.add_argument('-ct_npy_path', type=str, default='MRI-synthesized-from-CT/inference/ct_npy_path', help="Folder contains .npy ct images after extracting from DICOM format")
    # parser.add_argument('-prediction_path', type=str, default='MRI-synthesized-from-CT/inference/synthesized_MRI', help="Folder contains predictions")
    # parser.add_argument('-segmentation_path', type=str, default='MRI-synthesized-from-CT/inference/segmentations', help="Folder contains segmentations")
    parser.add_argument('-nnUNet_dir', type=str, default='./MRI-synthesized-from-CT/nnUNet', help='folder which stores nnUNet stuff')
    parser.add_argument('-device', type=str, default='cuda', help='device when using nnUNet')
    parser.add_argument('-save_dir', type=str, default='MRI-synthesized-from-CT/inference', help="Directory contains predictions and segmentations")
    
    args = parser.parse_args()
    
    inference_and_segmentation(args)
    
    