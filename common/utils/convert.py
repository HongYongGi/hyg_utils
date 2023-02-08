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
        
def 
