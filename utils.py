import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import random
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pydicom
import cv2
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import fid_score

# Cloning and downloading dependencies from my customized nnUNet's repository
# It's a bad way to use os.system(), it might be improved soon
# def nnUNet_download(): 
#     os.system('git clone https://github.com/LeTruongVu2k1/Customized-nnUNet.git')
#     os.system('cd Customized-nnUNet')
#     os.system('pip install -e ./Customized-nnUNet')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        
def loss_filter(mask,device="cuda"):
    list = []
    for i, m in enumerate(mask):
        if torch.any(m != 0):
            list.append(i)
    index = torch.tensor(list, dtype=torch.long).to(device)
    return index
        
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
      nets (network list)   -- a list of networks
      requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
# Use for plotting images in training loop
def show_tensor_images(image_tensor, num_images=25, size=(1, 256, 256)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor[0] = (image_tensor[0] + 1) / 2
    image_tensor = torch.cat([image_tensor[0], image_tensor[1]])
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=6)

    plt.figure(figsize=(12, 12))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
def nnUNet_predict(t2_fake, input_name, output_name):
    affine = np.eye(4)

    for i, img in enumerate(t2_fake):
        x = img.permute(1,2,0).numpy().copy()

        nii_file = nib.Nifti1Image(x, affine)

        nib.save(nii_file, f'input/{i}_0000.nii.gz')

    os.system(f'nnUNetv2_predict -d Dataset001_Liver -i {input_name} -o {output_name} -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans -device cuda')

    predictions = torch.zeros_like(t2_fake)

    for i in range(len(t2_fake)):
        predictions[i] = torch.from_numpy(nib.load(f'{output_name}/{i}.nii.gz').get_fdata()).permute(2,0,1)

    return predictions
    
def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def get_ct_img(path):
    dcm = pydicom.dcmread(path)
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

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

    img = (img - 0.5) / 0.5

    return torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0)

def raw_preprocess(data, get_s=False):
    """
    :param data: [155,224,224]
    :return:
    """
    data = data.astype(dtype=float)
    data[data<50] = 0
    out = data.copy()
    out = (out - out.min()) / (out.max() - out.min())
    # out = (out - 0.5) / 0.5
    if get_s:
        share_mask = out.copy()
        share_mask[share_mask != 0] = 1
        return out, share_mask
    return out

def get_t2(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)[0]

    img = raw_preprocess(img)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

    return img

def create_t2_fake(gen1, ct_batch, t2_fake_path, device='cuda'):
    gen1.eval()

    interval = len(ct_batch) // 26
    num_interval = 26
    s = 0
    with torch.inference_mode():
        for i in range(1,num_interval+1):
            with autocast():
                t2_fake_1_real = gen1(ct_batch[s:(interval*i)].to(device), mode='test').detach().cpu().numpy()
                t2_fake_1_real = t2_fake_1_real * 0.5 + 0.5

            for j in range(interval):
                cv2.imwrite(f'FID/{t2_fake_path}/{j+s}.png', cv2.cvtColor((t2_fake_1_real[j][0] * 255).round().astype(np.uint8),cv2.COLOR_GRAY2RGB))

            s = interval*i

        if s - 1 < len(ct_batch):
            with autocast():
                t2_fake_1_real = gen1(ct_batch[s:len(ct_batch)].to(device), mode='test').detach().cpu().numpy()
                t2_fake_1_real = t2_fake_1_real * 0.5 + 0.5

            for j in range(len(t2_fake_1_real)):
                cv2.imwrite(f'FID/{t2_fake_path}/{j+s}.png', cv2.cvtColor((t2_fake_1_real[j][0] * 255).round().astype(np.uint8),cv2.COLOR_GRAY2RGB))

    gen1.train()

def create_t2_real(t2_real, t2_real_path):
    for i in range(len(t2_real)):
        cv2.imwrite(f'FID/{t2_real_path}/{i}.png', cv2.cvtColor((t2_real[i] * 255).round().astype(np.uint8),cv2.COLOR_GRAY2RGB))


def calculate_FID(gen1, ct_batch, t2_real, dim, t2_fake_path='t2_fake', t2_real_path='t2_real'):
    os.system('rm -r FID')
    os.system(f'mkdir FID FID/{t2_fake_path} FID/{t2_real_path}')

    create_t2_fake(gen1, ct_batch, t2_fake_path)
    create_t2_real(t2_real, t2_real_path)

    # os.system(f'python -m pytorch_fid FID/{t2_fake_path} FID/{t2_real_path} --device cuda --dim {dim}')
    paths = [f'FID/{t2_fake_path}', f'FID/{t2_real_path}']
    device = 'cuda'
    fid_score.main(paths, dim, device)
    

