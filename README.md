# T2w-MR synthesizing based on CT 

This is my proposed model on translating abdominal slices from CT modality to T2w modality. This implementation is based on my research - *MR-synthesizing from CT using unpaired dataset*.

I evaluate my model's performance based on (1) [FID-score](https://arxiv.org/abs/1706.08500) and (2) [DICE](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) score on downstream segmentation task when augmenting the training data with the MRI synthesised by my model.

On downstream segmentation task (2), I use [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/master) to segment liver on MR slices.

<p align="center">
<img width="500" alt="model" src="images/cycle_consistency.png">
</p>

## Dataset
To download the CHAOS dataset from [the CHAOS challenge website](https://chaos.grand-challenge.org/Download/). 
Then, you need to create a folder structure like this:

    ├── train
    │   ├── ct
    │   │   ├──1
    │   │   ├──...(patient index)
    │   ├── t1
    │   │   ├──3
    │   │   ├──...(patient index)
    │   ├── t2
    │   │   ├──5
    │   │   ├──...(patient index)
    ├── test
    │   ├── ct
    │   │   ├──2
    │   │   ├──...(patient index)
    │   ├── t1
    │   │   ├──4
    │   │   ├──...(patient index)
    │   ├── t2
    │   │   ├──6
    │   │   ├──...(patient index)
    
## Installation
```bash
git clone https://github.com/LeTruongVu2k1/MRI-synthesized-from-CT
pip install -r MRI\ synthesis\ from\ CT/requirements.txt
```

## Training
Run the following script to reproduce results presented in our research:

```bash
python 'MRI synthesis from CT/train.py' -dataset_path <path_to_CHAOS_dataset> -model_dir <path_to_model> -nnUNet_dir <path_to_nnUNet_folder>
```

**NOTE**: If you want to run on *Colab*, due to this [problem](https://github.com/googlecolab/colabtools/issues/1067), please run like this:
```bash
%run 'MRI synthesis from CT/train.py' -dataset_path <path_to_CHAOS_dataset> -model_dir <path_to_model> -nnUNet_dir <path_to_nnUNet_folder>
```

## Inference
I also provide my pretrained model (100 epoch), which is stored in `checkpoint` dicrectory.

Run this script to synthesis MRI from your CT:

```bash
python 'MRI synthesis from CT/inference.py' -path <path_to_input_file>
```

You should structure the `path_to_input_file` similar to the test split in CHAOS dataset. Please note that you should contain both `DICOM_anon` and `Ground` folders, the `Ground` folder will be used for displaying:

    ├── path_to_input_file
    │   ├── ct
    │   │   ├──2
    │   │      ├── DICOM_anon
    │   │      ├── Ground
    │   │   ├──10
    │   │      ├── DICOM_anon
    │   │      ├── Ground
    │   │   ├──31
    │   │      ├── DICOM_anon
    │   │      ├── Ground
    │   │   ├──...(patient index)
    
The result will be stored in three folders:
- `inference/synthesized_MRI`

This folder will store the MRI synthesized from your CT input. <br>
The format will be: `inference/synthesized_MRI/<your_input_ct_name>_0000.nii.gz`

- `inference/segmentations`

This folder will store nnUNet's segmentation on your synthesized MRI. <br>
The format will be: `inference/segmentations/<your_input_ct_name>.nii.gz`

- `inference/ct_npy_path`

This folder will store your CT input and and the liver's mask. <br>
The ct image's format will be: `inference/ct_npy_path/<your_input_ct_name>.npy` <br>
The liver's mask's format will be: `inference/ct_npy_path/<your_liver_mask_name>.npy`

