'''
Log

* Written by YongGi Hong / email : hyg4438@gmail.com

*  Written date : 20230208
---

Description
* Convert Medical Image file function

'''

mport numpy as np
import pandas as pd
import nibabel as nib
import os,  glob, shutil
import dicom2nifti
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()
settings.disable_validate_siemens_slice_increment()
settings.disable_validate_siemens_slice_timing()
import nibabel as nib
import pydicom

from tqdm import tqdm
import ipywidgets as widgets
from datetime import date
###################################################################
def load_nii(nii_path):
    """ Load nii file """
    return nib.load(nii_path).get_fdata()

def load_dcm(dcm_dir):
    
    dcm_list = glob.glob(dcm_dir + '/*.dcm')
    dcm_list.sort()
    if len(dcm_list) == 0:
        dcm_list = glob.glob(dcm_dir + '/*.DCM')
        dcm_list.sort()
    dicoms = [pydicom.dcmread(dcm) for dcm in dcm_list]
    image = np.transpose([dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept for dicom in dicoms],
                         axes=(1, 2, 0))






###################################################################
def get_filename(path):
    """ Get the filename from a path"""
    return os.path.basename(path)

def get_dirname(path):
    """ Get the directory name from a path"""
    return os.path.dirname(path)

def makedir(dir):
    """ Create a directory if it does not exist"""
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def nii2niigz(nii_path):
    """ Convert nii to niigz """
    nii = nib.load(nii_path)
    nib.save(nii, nii_path + '.gz')
    os.remove(nii_path)


def niigz2nii(niigz_path):
    """ Convert niigz to nii """
    nii = nib.load(niigz_path)
    nib.save(nii, niigz_path[:-3])
    os.remove(niigz_path)


def dcm2nii(dcm_path, nii_path):
    """ 
    Convert dcm to nii 
    
    Args:
        dcm_path (str): dcm file path
        nii_path (str): nii file path
        
    """
    dicom2nifti.convert_directory(dcm_path, nii_path, compression=True, reorient=True)
    print('Convert Complete!')
    


