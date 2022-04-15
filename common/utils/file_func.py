# Path
import glob, os,shutil



# data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import json


# medical image 
import dicom2nifti.settings as settings
import dicom2nifti
from pydicom import read_file
import pydicom
import nibabel as nib


# etc
from tqdm import tqdm
import ipywidgets as widgets
from datetime import date

def remove_extension(file_path,file_slash='/',folder_flag=False):
    if folder_flag == False:
        split_name = file_path.split(file_slash)
#         print(split_name)
        file_name = split_name[-1]
        split_name.remove(file_name)
        join_name = file_slash.join(split_name)
        return join_name, file_name
    
    else : 
        split_name = file_path.split(file_slash)
#         print(split_name)
        file_name = split_name[-2]
        split_name.remove(file_name)
        join_name = file_slash.join(split_name)
        return join_name, file_name

def makedirs(path): 
    try: 
        os.makedirs(path) 
    except OSError: 
        if not os.path.isdir(path): 
            raise

            
            
            
            
def niigz2niigz_0000(filename):  
    if filename[-12:]!='_0000.nii.gz':
        os.rename(filename,filename[:-7]+'_0000.nii.gz')
    else :
        pass
    
    
    
    
def nii2niigz_0000(filename, output_dir, print_flag = False,file_slash='/'):
    '''
    file name : nii file path
    output_dir : converting niigz file save dir 
    '''
    
    
    folder_name, nii_file_name = remove_extension(filename,file_slash)
    nii_input = nib.load(filename)
    header = nii_input.header
    data = np.copy(nii_input.get_fdata())
    
    new_image = nib.Nifti1Image(data, nii_input.affine, header=header)
    save_path = os.path.join(output_dir,nii_file_name[:-4]+'_0000.nii.gz')
    nib.save(new_image, save_path)
    if print_flag==True:
        print(f"save the file : {nii_file_name}")
    else:
        pass
    return new_image




def nii2niigz(filename, output_dir, print_flag = False,file_slash='/'):
    '''
    file name : nii file path
    output_dir : converting niigz file save dir 
    '''
    
    
    folder_name, nii_file_name = remove_extension(filename,file_slash)
    nii_input = nib.load(filename)
    header = nii_input.header
    data = np.copy(nii_input.get_fdata())
    
    new_image = nib.Nifti1Image(data, nii_input.affine, header=header)
    save_path = os.path.join(output_dir,nii_file_name+'.gz')
    nib.save(new_image, save_path)
    if print_flag==True:
        print(f"save the file : {nii_file_name}")
    else:
        pass
    return new_image
    
    
def niigz2nii(filename, output_dir,print_flag=False,file_slash='/'):
    '''
    file name : niigz file path
    output_dir : converting nii file save dir 
    
    return value : 
    '''
    folder_name, nii_file_name = remove_extension(filename,file_slash)
    nii_input = nib.load(filename)
    header = nii_input.header
    data = np.copy(nii_input.get_fdata())
    
    new_image = nib.Nifti1Image(data, nii_input.affine, header=header)
    save_path = os.path.join(output_dir,nii_file_name[:-3])
    nib.save(new_image, save_path)
    if print_flag==True:
        print(f"save the file : {nii_file_name}")
    else:
        pass
    
    return new_image    
    
    
    
def load_ct(file_path):
    """ Loads in a CT dicom series and its corresponding weights

    """
    # read in dicom filenames
    dicom_files = glob.glob(os.path.join(file_path, '*.dcm'))
    dicoms = [read_file(dicom_file) for dicom_file in dicom_files]

    # sort dicoms by slices
    slice_sorts = np.argsort([dicom.SliceLocation for dicom in dicoms])
    dicoms = [dicoms[slice_sort] for slice_sort in slice_sorts]

    # read in pixel array
    image = np.transpose([dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept for dicom in dicoms],
                         axes=(1, 2, 0))

    # store slope/orientation to standardize image
    slope = np.float32(dicoms[1].ImagePositionPatient[2]) - \
            np.float32(dicoms[0].ImagePositionPatient[2])
    orientation = np.float32(dicoms[0].ImageOrientationPatient[4])

    # standardize image position
    if slope < 0:
        image = np.flip(image, -1)  # enforce feet first axially
    if orientation < 0:
        image = np.flip(image, 0)  # enforce supine orientation

    # extract voxel dimensions
    slice_space = np.abs(dicoms[1].SliceLocation - dicoms[0].SliceLocation)
    vox_dim = np.float32(dicoms[0].PixelSpacing[0]), np.float32(
        dicoms[0].PixelSpacing[1]), np.float32(slice_space)

    return image    
    
def save_nifti(dicom_path, save_path, filename, volume):
    dicom2nifti.dicom_series_to_nifti(dicom_path, save_path + 'temp.nii', reorient_nifti=False)
    temp = nib.load(save_path + 'temp.nii')
    temp_inform = temp.header
    temp_affine = temp.affine

    save_format = np.transpose(volume, (1,0,2))
    nifti = nib.Nifti1Image(save_format, temp_affine, temp_inform)
    nib.save(nifti, save_path + '/' + filename)

    os.remove(save_path + 'temp.nii')
    
    
    
    
    

    
def dcm2nii(input_dir, output_dir,file_slash='/'):
    '''
    input_dir : dcm folder(must finish '/')
    output_dir : save nii file directory
    
    '''
    parent_dir, nii_file_name = remove_extension(input_dir,file_slash,folder_flag=True)
    volume = load_ct(input_dir)
    file_name = nii_file_name+'.nii'
    save_nifti(input_dir,output_dir,file_name,volume)
    
    return volume
    
    
    
    
    
    
def dcm2niigz(input_dir, output_dir,input_flag =False):
    
    '''
    input_dir : dcm folder(must finish '/')
    output_dir : save nii file directory
    
    '''
    
    parent_dir, nii_file_name = remove_extension(input_dir,file_slash='/',folder_flag=True)
    volume = load_ct(input_dir)
    if input_flag==True:
        file_name = nii_file_name+'_0000.nii.gz'
    else: 
        file_name = nii_file_name+'.nii.gz'
    save_nifti(input_dir,output_dir,file_name,volume)
    return volume


import dicom2nifti

def dcm2niigz_v2(dicom_directory,output_directory,tmp_directory,file_slash='/'):
    parent_dir, nii_file_name = remove_extension(dicom_directory,file_slash ,folder_flag=True)
    
    dicom2nifti.convert_directory(dicom_directory, tmp_directory)
    tmp_name_niigz_file = glob.glob(tmp_directory+'*.nii.gz')[0]
    file_name = nii_file_name+'_0000.nii.gz'
    output_path = os.path.join(output_directory,file_name)
    os.rename(tmp_name_niigz_file,output_path)


###############################################################################################
def convertNsave(arr,file_dir,ref_dir, index=0):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put 
    the name of each slice while using a for loop to convert all the slices
    """
    dicom_f = glob.glob(ref_dir +'/*.dcm')
    dicom_file = pydicom.dcmread(dicom_f[0])
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.SeriesDescription = 'Research & Science make dicom data'
    dicom_file.PixelData = arr.tobytes()
    dicom_file.save_as(os.path.join(file_dir, f'slice{index}.dcm'))
    
    
def nifti2dicom_1file(nifti_dir, out_dir,ref):
    """
    This function is to convert only one nifti file into dicom series
    `nifti_dir`: the path to the one nifti file
    `out_dir`: the path to output
    """

    nifti_file = nib.load(nifti_dir)
    nifti_array = nifti_file.get_fdata()
    number_slices = nifti_array.shape[2]

    for slice_ in tqdm(range(number_slices)):
        convertNsave(nifti_array[:,:,slice_], out_dir, ref,slice_)  
        
        
def nii2dcm(nifti_dir, out_dir=''):
    """
    This function is to convert multiple nifti files into dicom files
    `nifti_dir`: You enter the global path to all of the nifti files here.
    `out_dir`: Put the path to where you want to save all the dicoms here.
    PS: Each nifti file's folders will be created automatically, so you do not need to create an empty folder for each patient.
    """

    files = os.listdir(nifti_dir)
    for file in files:
        in_path = os.path.join(nifti_dir, file)
        out_path = os.path.join(out_dir, file)
        os.mkdir(out_path)
        nifti2dicom_1file(in_path, out_path)
        
def niigz2dcm(nifti_dir, out_dir=''):
    """
    This function is to convert multiple nifti files into dicom files
    `nifti_dir`: You enter the global path to all of the nifti files here.
    `out_dir`: Put the path to where you want to save all the dicoms here.
    PS: Each nifti file's folders will be created automatically, so you do not need to create an empty folder for each patient.
    """

    files = os.listdir(nifti_dir)
    
    for file in files:
        in_path = os.path.join(nifti_dir, file)
        out_path = os.path.join(out_dir, file)
        os.mkdir(out_path)
        nifti2dicom_1file(in_path, out_path)
        
###############################################################################################

 
    
def nii2raw(niigz):
    '''
    niigz : nii 1 file
    '''
    
    
    img = nib.load(niigz)
    data = img.get_fdata()

    data = np.flip(data, axis=2)
    data = np.fliplr(data)
    data = np.flipud(data)

    unique_val = np.unique(data)

    for unq in unique_val[1:]:
        mask = np.where(data==unq, 1, 0)
        fileobj = open(niigz[:-4] + f'_gt{int(unq)}.raw', mode='wb')
        off = np.array(np.transpose(mask, axes=[2, 1, 0]), dtype=np.uint8)
        off.tofile(fileobj)
        fileobj.close()
    print(niigz[:-4], 'saved')
    
    
    

def niigz2raw(niigz):
    img = nib.load(niigz)
    data = img.get_fdata()

    data = np.flip(data, axis=2)
    data = np.fliplr(data)
    data = np.flipud(data)

    unique_val = np.unique(data)

    for unq in unique_val[1:]:
        mask = np.where(data==unq, 1, 0)
        fileobj = open(niigz[:-7] + f'_gt{int(unq)}.raw', mode='wb')
        off = np.array(np.transpose(mask, axes=[2, 1, 0]), dtype=np.uint8)
        off.tofile(fileobj)
        fileobj.close()
    print(niigz[:-7], 'saved')


def raw2nii(raw_dir, raw_file_name_condition, nii_file_path):
    
    '''
    ex ) nii_path = './data/case1.nii'
    raw_dir = './gt/'
    raw_file_name_condition = 'case1'
    
    
    


    '''
    nii_input = nib.load(nii_file_path)
    header = nii_input.header
    
    
    raw_files = glob.glob(os.path.join(raw_dir,f'{raw_file_name_condition}*.raw'))
    
    mask_out = np.zeros((nii_input.shape[2], nii_input.shape[1], nii_input.shape[0]))
    
    for i, raw_file in enumerate(raw_files,1):
        cls = int(raw_file[raw_file.rfind('_out')+4:raw_file.rfind('.')])
        mask = np.fromfile(raw_file, dtype='uint8', sep="")
        mask = mask.reshape(mask_out.shape)
        mask_out = np.where(mask, cls, mask_out)
        print(raw_file, cls)
    
    nii = nib.Nifti1Image(np.transpose(mask_out.astype(np.int16), axes=[2, 1, 0]), affine=None, header=header)
    print(nii_file_path[-3:])
    if nii_file_path[-3:] == '.gz':
        nib.save(nii,nii_file_path[:-7]+'_gt.nii')
        print(nii_file_path[:-7])
    else : 

        nib.save(nii, nii_file_path[:-4]+'_gt.nii')
        print(nii_file_path[:-4])


    
    
    
def raw2niigz(raw_dir, raw_file_name_condition, nii_file_path):
    
    
    '''
    ex ) nii_path = './data/case1.nii'
    raw_dir = './gt/'
    raw_file_name_condition = 'case1'
    
    
    


    '''
    nii_input = nib.load(nii_file_path)
    header = nii_input.header
    
    
    raw_files = glob.glob(os.path.join(raw_dir,f'{raw_file_name_condition}*.raw'))
    
    mask_out = np.zeros((nii_input.shape[2], nii_input.shape[1], nii_input.shape[0]))
    
    for i, raw_file in enumerate(raw_files,1):
        cls = int(raw_file[raw_file.rfind('_out')+4:raw_file.rfind('.')])
        mask = np.fromfile(raw_file, dtype='uint8', sep="")
        mask = mask.reshape(mask_out.shape)
        mask_out = np.where(mask, cls, mask_out)
        print(raw_file, cls)
    
    nii = nib.Nifti1Image(np.transpose(mask_out.astype(np.int16), axes=[2, 1, 0]), affine=None, header=header)
    print(nii_file_path[-3:])
    if nii_file_path[-3:] == '.gz':
        nib.save(nii,nii_file_path[:-7]+'_gt.nii.gz')
        print(nii_file_path[:-7])
    else : 

        nib.save(nii, nii_file_path[:-4]+'_gt.nii.gz')
        print(nii_file_path[:-4])


   
