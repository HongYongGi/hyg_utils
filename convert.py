'''
Log

* Written by YongGi Hong / email : hyg4438@gmail.com

*  Written date : 20230208
---

Description
* Convert Medical Image file function

'''

import numpy as np
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
    return image


    




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

def nii2dcm(save_dir_path, nifti_dir_path, ref_dicom_dir_path, debug = False):

    """ Convert nii to dcm
    
    Parameters
    ----------
    save_dir_path : str 
    nifti_dir_path : str
    ref_dicom_dir_path : str
    debug : bool
    If True, print the path of nifti and dicom files


    """
    nii_paths = glob.glob(nifti_dir_path + '/*.nii')
    nii_paths.sort()
    dicom_dir_paths = glob.glob(ref_dicom_dir_path + '/*/')
    dicom_dir_paths.sort()

    # nifti files, reference dicom directories check the number
    if len(nii_paths)!=len(dicom_dir_paths):
        print('The number of nii files and dicom directories are not the same.')

    else: 
        for nii_path, dicom_dir_path in tqdm(zip(nii_paths, dicom_dir_paths)):
            nii = load_nii(nii_path)
            dcm = load_dcm(dicom_dir_path)
            if nii.shape != dcm.shape:
                print('The shape of nii and dicom are not the same.')
                print('nii shape : ', nii.shape)
                print('dcm shape : ', dcm.shape)
                break
            else : 
                pass
            

            save_dir   =os.path.join(save_dir_path, get_dirname(nii_path))
            os.makedirs(save_dir, exist_ok=True)
            dicom_file_paths = glob.glob(dicom_dir_path + '/*.dcm')
            dicom_file_paths.sort()
            if len(dicom_file_paths) == 0:
                dicom_file_paths = glob.glob(dicom_dir_path + '/*.DCM')
                dicom_file_paths.sort()
            
            nifti_array = nii
            nifti_array = np.flip(np.flip(nifti_array,1),2)
            nifti_array = np.swapaxes(nifti_array, 0, 1)

            nifti_slices = nifti_array.shape[2]

            for slice_idx in range(nifti_slices):
                nii_2d_array= nifti_array[:,:,slice_idx]
                dicom_file = pydicom.dcmread(dicom_file_paths[slice_idx])
                arr = nii_2d_array.astype(np.int16)
                dicom_file.Rows, dicom_file.Columns = arr.shape
                dicom_file.PixelData = arr.tobytes()
                dicom_file.BitsStored = 16
                dicom_file.BitsAllocated = 16
                dicom_file.HighBit = 15
                dicom_file.PixelRepresentation = 0
                dicom_file.RescaleIntercept = 0
                dicom_file.RescaleSlope = 1
                dicom_file.PixelSpacing = [dicom_file.PixelSpacing[0], dicom_file.PixelSpacing[1]]
                dicom_file.SeriesDescription = 'Converted from Nifti file'
                dicom_file.save_as(os.path.join(save_dir,f'slice_00{slice:04}.dcm'))
    clear_output(wait=True)
    print('nifti2dicom conversion completed!')



