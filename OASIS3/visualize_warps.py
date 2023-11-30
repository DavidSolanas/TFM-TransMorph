import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import scipy.io

if __name__ == '__main__':

    warp_data_ncc = nib.load('C:/Users/david/Documents/Master Ingenieria Informatica/TFM/dataset/InfersCycleGAN+Voxelmorph_NCC_warps/OAS30030_d0170-OAS30051_d0200_warp.nii.gz').get_fdata()
    warp_data_mi = nib.load('C:/Users/david/Documents/Master Ingenieria Informatica/TFM/dataset/InfersCycleGAN+Voxelmorph_MI_warps/OAS30030_d0170-OAS30051_d0200_warp.nii.gz').get_fdata()
    warp_data_mse = nib.load('C:/Users/david/Documents/Master Ingenieria Informatica/TFM/dataset/InfersCycleGAN+Voxelmorph_MSE_warps/OAS30030_d0170-OAS30051_d0200_warp.nii.gz').get_fdata()
    warp_data_synthseg = nib.load('C:/Users/david/Documents/Master Ingenieria Informatica/TFM/dataset/OASIS3_final_SynthReg/test/3_warp.nii').get_fdata()
    warp_data_nirep = nib.load('C:/Users/david/Documents/Master Ingenieria Informatica/TFM/code/matlab/Jacobians/StationaryLDDMM_Metrics_2021/MultiResolution/NCC/OAS30030_d0170-OAS30051_d0200_warp.nii').get_fdata()

    ##################################################################
    # SYNTHSEG
    print(warp_data_synthseg.shape)
    
    # Get the indices for the central slices along each axis
    x_slice = warp_data_synthseg.shape[2] // 2

    # Extract the vectors for the central slices
    x_synthseg_slice_data = warp_data_synthseg[:, :, x_slice, :]
    
    
    # Assuming your image is a NumPy array of shape (128, 128, 3)
    # You can replace this with your actual image data
    image_synthseg_data = x_synthseg_slice_data

    # Assuming your 3D vectors are stored in a NumPy array with shape (128, 128, 3)

    # Separate the components
    x_component = image_synthseg_data[:, :, 0]
    y_component = image_synthseg_data[:, :, 1]
    z_component = image_synthseg_data[:, :, 2]

    # Normalize components to [0, 1]
    x_normalized = (x_component - x_component.min()) / (x_component.max() - x_component.min())
    y_normalized = (y_component - y_component.min()) / (y_component.max() - y_component.min())
    z_normalized = (z_component - z_component.min()) / (z_component.max() - z_component.min())

    # Create RGB colors
    colors_synthseg = np.stack([x_normalized, y_normalized, z_normalized], axis=-1)
    ##########################################################################


    #########################################################################
    # VXM NCC
    print(warp_data_synthseg.shape)
    
    # Get the indices for the central slices along each axis
    x_slice = warp_data_ncc.shape[2] // 2

    # Extract the vectors for the central slices
    x_slice_data = warp_data_ncc[:, :, x_slice, :]
    
    
    # Assuming your image is a NumPy array of shape (128, 128, 3)
    # You can replace this with your actual image data
    image_data = x_slice_data

    # Assuming your 3D vectors are stored in a NumPy array with shape (128, 128, 3)

    # Separate the components
    x_component = image_data[:, :, 0]
    y_component = image_data[:, :, 1]
    z_component = image_data[:, :, 2]

    # Normalize components to [0, 1]
    x_normalized = (x_component - x_component.min()) / (x_component.max() - x_component.min())
    y_normalized = (y_component - y_component.min()) / (y_component.max() - y_component.min())
    z_normalized = (z_component - z_component.min()) / (z_component.max() - z_component.min())

    # Create RGB colors
    colors_ncc = np.stack([x_normalized, y_normalized, z_normalized], axis=-1)

    #########################################################################


    #########################################################################
    # VXM MI
    # Get the indices for the central slices along each axis
    x_slice = warp_data_mi.shape[2] // 2

    # Extract the vectors for the central slices
    x_slice_data = warp_data_mi[:, :, x_slice, :]
    
    
    # Assuming your image is a NumPy array of shape (128, 128, 3)
    # You can replace this with your actual image data
    image_data = x_slice_data

    # Assuming your 3D vectors are stored in a NumPy array with shape (128, 128, 3)

    # Separate the components
    x_component = image_data[:, :, 0]
    y_component = image_data[:, :, 1]
    z_component = image_data[:, :, 2]

    # Normalize components to [0, 1]
    x_normalized = (x_component - x_component.min()) / (x_component.max() - x_component.min())
    y_normalized = (y_component - y_component.min()) / (y_component.max() - y_component.min())
    z_normalized = (z_component - z_component.min()) / (z_component.max() - z_component.min())

    # Create RGB colors
    colors_mi = np.stack([x_normalized, y_normalized, z_normalized], axis=-1)

    #########################################################################


    #########################################################################
    # VXM MSE
    x_slice = warp_data_mse.shape[2] // 2

    # Extract the vectors for the central slices
    x_slice_data = warp_data_mse[:, :, x_slice, :]
    
    
    # Assuming your image is a NumPy array of shape (128, 128, 3)
    # You can replace this with your actual image data
    image_data = x_slice_data

    # Assuming your 3D vectors are stored in a NumPy array with shape (128, 128, 3)

    # Separate the components
    x_component = image_data[:, :, 0]
    y_component = image_data[:, :, 1]
    z_component = image_data[:, :, 2]

    # Normalize components to [0, 1]
    x_normalized = (x_component - x_component.min()) / (x_component.max() - x_component.min())
    y_normalized = (y_component - y_component.min()) / (y_component.max() - y_component.min())
    z_normalized = (z_component - z_component.min()) / (z_component.max() - z_component.min())

    # Create RGB colors
    colors_mse = np.stack([x_normalized, y_normalized, z_normalized], axis=-1)

    #########################################################################

    #########################################################################
    # VXM NIREP
    x_slice = warp_data_nirep.shape[2] // 2

    # Extract the vectors for the central slices
    x_slice_data = warp_data_nirep[:, :, x_slice, :]
    
    
    # Assuming your image is a NumPy array of shape (128, 128, 3)
    # You can replace this with your actual image data
    image_data = x_slice_data

    # Assuming your 3D vectors are stored in a NumPy array with shape (128, 128, 3)

    # Separate the components
    x_component = image_data[:, :, 0]
    y_component = image_data[:, :, 1]
    z_component = image_data[:, :, 2]

    # Normalize components to [0, 1]
    x_normalized = (x_component - x_component.min()) / (x_component.max() - x_component.min())
    y_normalized = (y_component - y_component.min()) / (y_component.max() - y_component.min())
    z_normalized = (z_component - z_component.min()) / (z_component.max() - z_component.min())

    # Create RGB colors
    colors_nirep = np.stack([x_normalized, y_normalized, z_normalized], axis=-1)

    #########################################################################




    # Create a figure with 1 row and 4 columns
    fig, axs = plt.subplots(1, 5, figsize=(12, 3))

    images = [colors_nirep, colors_synthseg, colors_ncc, colors_mi, colors_mse]
    titles = ['NIREP', 'Synthseg', 'CycleGAN+Vxm NCC', 'CycleGAN+Vxm MI', 'CycleGAN+Vxm MSE']

    # Display each image in a separate subplot
    for i in range(5):
        axs[i].imshow(images[i])
        axs[i].set_title(titles[i])
        axs[i].axis('off')  # Turn off axis labels

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Show the plot
    plt.show()



