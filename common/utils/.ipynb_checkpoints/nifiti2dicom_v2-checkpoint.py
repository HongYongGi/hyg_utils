
import os
import numpy as np
import pydicom
import nibabel
from tqdm import tqdm


def nifti2dicom_v2_monib(root_path, nifti_dir_path, ref_dicom_dir_path, debug= False):
    """
    params:
    nifti_dir_path: path to the dir where nifti files are stored
    ref_dicom_dir_path: path to reference dicom dir <- reference dicom dir is the dicom files that nifti files are made from
    save_dicom_path: path to save the converted nifti to dicom files

    Note: For each nifti file, there is one dicom dir.
    Note2: Read the corresponding meta data from each ref dicom data and store the nifti array to that dicom file

    """

    path_nifti_files = [os.path.join(nifti_dir_path, file_name) for file_name in os.listdir(nifti_dir_path)]

    # reference dicom files for each nifti files
    dicom_dir_paths = [os.path.join(ref_dicom_dir_path, single_dicom_dir) for single_dicom_dir in
                       (os.listdir(ref_dicom_dir_path))]
    path_dicom_files = [os.path.join(dicom_dir, os.listdir(dicom_dir)[0]) for dicom_dir in dicom_dir_paths]


    for i in tqdm(range(len(path_nifti_files))):
        if debug:
            print(f"{path_nifti_files[i]}, {path_dicom_files[i]}")

        save_path = path_nifti_files[i].strip('.nii').split('/')[-1]
        save_path = os.path.join(root_path, f"nii2dicom/{save_path}")

        os.makedirs(save_path, exist_ok=True)

        # read nifti file
        nifti_file = nibabel.load(path_nifti_files[i])
        nifti_array = nifti_file.get_fdata()
        nifti_array = np.swapaxes(nifti_array, 0, 1)

        nifti_slices = nifti_array.shape[2]

        for slice in range(nifti_slices):
            nifty_2d_array = nifti_array[:, :, slice]

            dicom_file = pydicom.dcmread(path_dicom_files[i])
            arr = nifty_2d_array.astype('uint16')
            dicom_file.Rows = arr.shape[0]
            dicom_file.Columns = arr.shape[1]
            dicom_file.PhotometricInterpretation = "MONOCHROME2"
            dicom_file.SamplesPerPixel = 1
            dicom_file.BitsStored = 16
            dicom_file.BitsAllocated = 16
            dicom_file.HighBit = 15
            dicom_file.PixelRepresentation = 1
            dicom_file.SeriesDescription = "R&S Data"
            dicom_file.PixelData = arr.tobytes()
            dicom_file.save_as(os.path.join(save_path, f'slice_00{slice:04}.dcm'))

    print('nifti2dicom conversion completed!')

if __name__ == "__main__":
    pass