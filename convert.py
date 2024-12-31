'''
Log

* Written by YongGi Hong / email : hyg4438@gmail.com
*  Written date : 20241220
---

Description
* Convert Medical Image file function

'''

import numpy as np
import pandas as pd
import nibabel as nib
import os,  glob, shutil
import dicom2nifti
import nibabel as nib
import pydicom

from tqdm import tqdm
import ipywidgets as widgets
from datetime import date


###################################################################
def load_nii(nii_path):
    """ Load nii file """
    return nib.load(nii_path).get_fdata()

def save_nii(path, header, affine, arr):
    nii = nib.Nifti1Image(arr, affine, header)
    nib.save(nii, path)

def load_dcm(dcm_dir):
    
    dcm_list = glob.glob(dcm_dir + '/*.dcm')
    dcm_list.sort()
    if len(dcm_list) == 0:
        dcm_list = glob.glob(dcm_dir + '/*.DCM')
        dcm_list.sort()
    dicoms = [pydicom.dcmread(dcm) for dcm in dcm_list]
    image = np.transpose([dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept for dicom in dicoms],
                         axes=(1, 2, 0))
    return image

def dcm2nii(dcm_dir, nii_path):
    dicom2nifti.convert_directory(dcm_dir, nii_path)
    nii = nib.load(nii_path)
    header = nii.header
    affine = nii.affine
    arr = load_dcm(dcm_dir)
    nii_image = nib.Nifti1Image(arr, affine, header)
    nib.save(nii_image, nii_path)

def nii2niigz(nii_path):
    nii = nib.load(nii_path)
    affine = nii.affine
    header = nii.header
    arr = nii.get_fdata()
    save_path = nii_path.replace('.nii', '.nii.gz')
    save_nii(nii_path, header, affine, arr)

def niigz2nii(niigz_path):
    nii = nib.load(niigz_path)
    affine = nii.affine
    header = nii.header
    arr = nii.get_fdata()
    save_path = niigz_path.replace('.nii.gz', '.nii')
    save_nii(save_path, header, affine, arr)

def nnunet_preprocess(nii_path):
    nii = nib.load(nii_path)
    header = nii.header
    affine = nii.affine
    arr = nii.get_fdata()
    file_name = os.path.basename(nii_path)
    if file_name.endswith('.nii.gz'):
        file_name = file_name.replace('.nii.gz', '_0000.nii.gz')
    elif file_name.endswith('.nii'):
        file_name = file_name.replace('.nii', '_0000.nii.gz')
    else:
        raise ValueError(f"Invalid file name: {file_name}")
    
    save_path = os.path.join(os.path.dirname(nii_path), file_name)
    save_nii(save_path, header, affine, arr)
    return save_path
