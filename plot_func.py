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


###################################################


def read_data(path):
    if path[-4:]=='json':
        with open(path, "r", encoding="utf8") as f: 
            contents = f.read() # string 타입 
            data = json.loads(contents)

    elif path[-4:] == '.pkl':
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
    
    
    elif path[-4:] == '.nii':
        data = nib.load(path)
        data = data.get_fdata()
    
    elif path[-4:] == 'i.gz':
        data = nib.load(path)
        data = data.get_fdata()
    
    else : 
        print("Not available read file")
        data = []
        
    return data
















def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)







def image_plot(nii_path , select_x=256,select_y=256,select_z=33):
    data_path = nii_path
    data = read_data(data_path)
    x = select_x
    y= select_y
    z = select_z

    fig = plt.figure(figsize= (100,100))
    ax1 = fig.add_subplot(131)
    ax1.imshow(np.rot90(data[:,:,z],3),cmap='gray')
    forceAspect(ax1,1)
    plt.axis('off')

    ax2 = fig.add_subplot(132)
    ax2.imshow(np.rot90(data[:,y,:],1),cmap='gray')
    forceAspect(ax2,1)
    plt.axis('off')



    ax3 = fig.add_subplot(133)
    ax3.imshow(np.rot90(data[x,:,:],1),cmap='gray')
    forceAspect(ax3,1)
    plt.axis('off')


    
def comparison_image_plot(nii_path,
                          target_path, 
                          mask_flag=False,
                          mask_number = 1,
                          blending_ratio = 0.1,
                          select_x=256,
                          select_y=256,
                          select_z=33, 
                          select_plot = False,
                          select_index = 1):
    
    data = read_data(nii_path)
    mask = read_data(target_path)
    

    x = select_x
    y= select_y
    z = select_z
    
    
    if mask_flag == True: 
        zeros = np.zeros(mask.shape)
        zeros[mask==mask_number]=1
        mask = zeros
        
    else :
        pass
    

    if select_plot == True : 
        fig = plt.figure(figsize= (100,100))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        
        
        if   select_index == 1 :
            a = np.rot90(data[:,:,z],3)
            b = np.rot90(mask[:,:,z],3)
            c = np.rot90(data[:,:,z]*(blending_ratio),3) + np.rot90(mask[:,:,z]*800*(1-blending_ratio),3)
            
            
        elif select_index ==2 :
            a = np.rot90(data[:,y,:])
            b = np.rot90(mask[:,y,:])
            c = np.rot90(data[:,y,:]*(blending_ratio)) + np.rot90(mask[:,y,:]*800*(1-blending_ratio))
            
        
        else : 
            a = np.rot90(data[x,:,:])
            b = np.rot90(mask[x,:,:])
            c = np.rot90(data[x,:,:]*(blending_ratio)) + np.rot90(mask[x,:,:]*800*(1-blending_ratio))
        ax1.imshow(a,cmap='gray')
        ax2.imshow(b,cmap='gray')
        ax3.imshow(c,cmap='gray')
        forceAspect(ax1,1)
        forceAspect(ax2,1)
        forceAspect(ax3,1)
        plt.axis('off')


        ax1.set_title("input", size=60)
        ax2.set_title("predict", size=60)
        ax3.set_title("blending", size=60)
    else : 
    
    
    


        fig = plt.figure(figsize= (100,100))
        ax1 = fig.add_subplot(331)
        ax1.imshow(np.rot90(data[:,:,z],3),cmap='gray')
        forceAspect(ax1,1)
        plt.axis('off')



        ax2 = fig.add_subplot(332)
        ax2.imshow(np.rot90(mask[:,:,z],3),cmap='gray')

        ax2.set_xlabel('xlabel')
        forceAspect(ax2,1)
        plt.axis('off')






        ax3 = fig.add_subplot(333)
        ax3.imshow(np.rot90(data[:,:,z]*(blending_ratio),3) + np.rot90(mask[:,:,z]*800*(1-blending_ratio),3),cmap='gray')

        ax3.set_xlabel('xlabel')
        forceAspect(ax3,1)
        plt.axis('off')


        ax4 = fig.add_subplot(334)
        ax4.imshow(np.rot90(data[:,y,:]),cmap='gray')

        ax4.set_xlabel('xlabel')
        forceAspect(ax4,1)
        plt.axis('off')



        ax5 = fig.add_subplot(335)
        ax5.imshow(np.rot90(mask[:,y,:]),cmap='gray')

        ax5.set_xlabel('xlabel')
        forceAspect(ax5,1)
        plt.axis('off')


        ax6 = fig.add_subplot(336)
        ax6.imshow(np.rot90(data[:,y,:]*(blending_ratio)) + np.rot90(mask[:,y,:]*800*(1-blending_ratio)),cmap='gray')

        ax6.set_xlabel('xlabel')
        forceAspect(ax6,1)
        plt.axis('off')



        ax7 = fig.add_subplot(337)
        ax7.imshow(np.rot90(data[x,:,:]),cmap='gray')

        ax7.set_xlabel('xlabel')
        forceAspect(ax7,1)
        plt.axis('off')



        ax8 = fig.add_subplot(338)
        ax8.imshow(np.rot90(mask[x,:,:]),cmap='gray')

        ax8.set_xlabel('xlabel')
        forceAspect(ax8,1)
        plt.axis('off')








        ax9 = fig.add_subplot(339)
        ax9.imshow(np.rot90(data[x,:,:]*(blending_ratio)) + np.rot90(mask[x,:,:]*800*(1-blending_ratio)),cmap='gray')

        ax9.set_xlabel('xlabel')
        forceAspect(ax9,1)
        plt.axis('off')


        ax1.set_title("input", size=60)
        ax2.set_title("predict", size=60)
        ax3.set_title("blending", size=60)


        ax4.set_title("input", size=60)
        ax5.set_title("predict", size=60)
        ax6.set_title("blending", size=60)



        ax7.set_title("input", size=60)
        ax8.set_title("predict", size=60)
        ax9.set_title("blending", size=60)







class DrawMaskSample:
    def __init__(self, nii_paths, i, ax):
        self.xyz_li = None
        self.i = i
        self.ax = ax
        self.nii_sample = nib.load(nii_paths[self.i]).get_fdata()
        self.get_xyz()
        self.draw_sample_3d()

    def get_xyz(self):
        self.xyz_li = []
        cnt = 0
        max_cnt = self.nii_sample.shape[-1]
        for iter_z, (iter_img) in enumerate(self.nii_sample.transpose(2, 0, 1)):
            for iter_x, iter_arr_y in enumerate(iter_img):
                iter_arr_y = np.where(iter_arr_y)[0]
                if len(iter_arr_y) >= 1:
                    iter_arr_y = list(
                        {iter_arr_y.max(), iter_arr_y.min()})
                    xyz = [
                        (iter_x, iter_y, iter_z)
                        for iter_y in iter_arr_y
                        if np.any(iter_y)
                    ]
                    self.xyz_li.append(xyz)
            cnt += 1

    def draw_sample_3d(self):
        xyz_matrix = np.array(list(itertools.chain.from_iterable(self.xyz_li)))
        X = xyz_matrix[:, 0]
        Y = xyz_matrix[:, 1]
        Z = xyz_matrix[:, 2]

        self.ax.scatter(X, Y, Z, s=1, alpha=0.05, color='#e3dac9')
        xlim, ylim, zlim = self.nii_sample.shape
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(0, ylim)
        self.ax.set_zlim(0, zlim)
        self.ax.set_title(f'sample - ({self.i + 1})')
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        self.ax.set_zticklabels([])
        self.ax.set_facecolor('#081921')



# fig = plt.figure(figsize=(24, 24))
# for i in range(9):
#     ax = fig.add_subplot(int(f'33{i + 1}'), projection='3d')
#     DrawMaskSample(nii_paths, i, ax)

# plt.tight_layout()
# plt.show()