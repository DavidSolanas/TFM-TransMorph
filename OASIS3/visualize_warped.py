import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import scipy.io

if __name__ == '__main__':

    warp_data_ncc = nib.load('C:/Users/david/Documents/Master Ingenieria Informatica/TFM/dataset/InfersCycleGAN+Voxelmorph_NCC_warps/OAS30030_d0170-OAS30051_d0200_warped.nii.gz').get_fdata()
    warp_data_mi = nib.load('C:/Users/david/Documents/Master Ingenieria Informatica/TFM/dataset/InfersCycleGAN+Voxelmorph_MI_warps/OAS30030_d0170-OAS30051_d0200_warped.nii.gz').get_fdata()
    warp_data_mse = nib.load('C:/Users/david/Documents/Master Ingenieria Informatica/TFM/dataset/InfersCycleGAN+Voxelmorph_MSE_warps/OAS30030_d0170-OAS30051_d0200_warped.nii.gz').get_fdata()
    warp_data_synthseg = nib.load('C:/Users/david/Documents/Master Ingenieria Informatica/TFM/dataset/OASIS3_final_SynthReg/test/3_warped.nii').get_fdata()
    warp_data_nirep = nib.load('C:/Users/david/Documents/Master Ingenieria Informatica/TFM/code/matlab/Jacobians/StationaryLDDMM_Metrics_2021/MultiResolution/NCC/OAS30030_d0170-OAS30051_d0200_warped.nii').get_fdata()

    ##################################################################
    # SYNTHSEG
    print(warp_data_synthseg.shape)
    
    # Get the indices for the central slices along each axis
    x_slice = warp_data_synthseg.shape[2] // 2

    # Extract the vectors for the central slices
    x_synthseg_slice_data = warp_data_synthseg[:, :, x_slice]
    
    
    # Assuming your image is a NumPy array of shape (128, 128, 3)
    # You can replace this with your actual image data
    image_synthseg_data = x_synthseg_slice_data

    # Assuming your 3D vectors are stored in a NumPy array with shape (128, 128, 3)


    # Create RGB colors
    colors_synthseg = image_synthseg_data
    ##########################################################################


    #########################################################################
    # VXM NCC
    print(warp_data_synthseg.shape)
    
    # Get the indices for the central slices along each axis
    x_slice = warp_data_ncc.shape[2] // 2

    # Extract the vectors for the central slices
    x_slice_data = warp_data_ncc[:, :, x_slice]
    
    
    # Assuming your image is a NumPy array of shape (128, 128, 3)
    # You can replace this with your actual image data
    image_data = x_slice_data

    # Assuming your 3D vectors are stored in a NumPy array with shape (128, 128, 3)


    # Create RGB colors
    colors_ncc = image_data

    #########################################################################


    #########################################################################
    # VXM MI
    # Get the indices for the central slices along each axis
    x_slice = warp_data_mi.shape[2] // 2

    # Extract the vectors for the central slices
    x_slice_data = warp_data_mi[:, :, x_slice]
    
    
    # Assuming your image is a NumPy array of shape (128, 128, 3)
    # You can replace this with your actual image data
    image_data = x_slice_data

    # Assuming your 3D vectors are stored in a NumPy array with shape (128, 128, 3)

    # Create RGB colors
    colors_mi = image_data

    #########################################################################


    #########################################################################
    # VXM MSE
    x_slice = warp_data_mse.shape[2] // 2

    # Extract the vectors for the central slices
    x_slice_data = warp_data_mse[:, :, x_slice]
    
    
    # Assuming your image is a NumPy array of shape (128, 128, 3)
    # You can replace this with your actual image data
    image_data = x_slice_data


    # Create RGB colors
    colors_mse = image_data

    #########################################################################

    #########################################################################
    # VXM NIREP
    x_slice = warp_data_nirep.shape[2] // 2

    # Extract the vectors for the central slices
    x_slice_data = warp_data_nirep[:, :, x_slice]
    
    
    # Assuming your image is a NumPy array of shape (128, 128, 3)
    # You can replace this with your actual image data
    image_data = x_slice_data

    # Assuming your 3D vectors are stored in a NumPy array with shape (128, 128, 3)


    # Create RGB colors
    colors_nirep = image_data

    #########################################################################




    # Create a figure with 1 row and 4 columns
    fig, axs = plt.subplots(1, 5, figsize=(12, 3))

    images = [colors_nirep, colors_synthseg, colors_ncc, colors_mi, colors_mse]
    titles = ['NIREP', 'Synthseg', 'CycleGAN+Vxm NCC', 'CycleGAN+Vxm MI', 'CycleGAN+Vxm MSE']

    # Display each image in a separate subplot
    for i in range(5):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')  # Turn off axis labels

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Show the plot
    plt.show()



