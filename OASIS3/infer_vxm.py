import glob
import os, sys
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
from torchvision import transforms
from natsort import natsorted
import neurite as ne
import matplotlib.pyplot as plt

# os.getcwd() is assumed to be in a folder inside the root git repo folder
code_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(code_folder)
sys.path.append(code_folder + '/Baseline_registration_models')
sys.path.append(code_folder + '/Baseline_registration_models/VoxelMorph')
sys.path.append(code_folder + '/OASIS')
sys.path.append(code_folder + '/OASIS/TransMorph')
sys.path.append(code_folder + '/OASIS/TransMorph/data')

from data import trans, datasets
from Baseline_registration_models.VoxelMorph import utils, losses, models
from OASIS.TransMorph import utils as utils_OASIS
from Baseline_registration_models.VoxelMorph.train_vxm import adjust_learning_rate, mk_grid_img, comput_fig, save_checkpoint, Logger

def plot_result(moving, fixed, warped, outputpath, x_patient_id, y_patient_id):
    # Check if the directory already exists
    if not os.path.exists(f"{outputpath}/{x_patient_id}-{y_patient_id}"):
        # Create the directory
        os.makedirs(f"{outputpath}/{x_patient_id}-{y_patient_id}")
    
    img = moving - fixed

    images = [img[0, :, :, 64], img[0, :, 64, :], img[0, 64, :, :]]

    # visualize
    #ne.plot.slices(images, cmaps=['gray'], do_colorbars=True, titles=['Moving - fixed']*3)
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    slice_labels=['Moving - fixed']*3
    # Loop through the neurite slices and plot each slice in a subplot
    for i, slice_data in enumerate(images):
        axs[i].imshow(slice_data, cmap='gray')
        axs[i].set_title(slice_labels[i])
        axs[i].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure as a single image
    save_path = f"{outputpath}/{x_patient_id}-{y_patient_id}/moving_fixed.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    img = warped - fixed

    images = [img[0, :, :, 64], img[0, :, 64, :], img[0, 64, :, :]]

    # visualize
    #ne.plot.slices(images, cmaps=['gray'], do_colorbars=True, titles=['Warped - fixed']*3)
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    slice_labels=['Warped - fixed']*3
    # Loop through the neurite slices and plot each slice in a subplot
    for i, slice_data in enumerate(images):
        axs[i].imshow(slice_data, cmap='gray')
        axs[i].set_title(slice_labels[i])
        axs[i].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure as a single image
    save_path = f"{outputpath}/{x_patient_id}-{y_patient_id}/warped_fixed.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main():
    ROOT_DATA_DIR = '/Disco2021-I/david/tfm/dataset/OASIS3_processed/'
    ROOT_LOG_DIR = '/Disco2021-I/david/tfm/tfm-tests/oasis3_500_pairs/'
    
    test_dir = ROOT_DATA_DIR + 'test/'
    model_folder = 'OASIS3_500_pairs_MSE/'
    model_idx = -1
    vol_size = (128, 128, 128) # resize img to half the original size
    model_dir = ROOT_LOG_DIR + 'experiments/' + model_folder
    
    model = models.VxmDense_1(vol_size)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir), reverse=True)[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir), reverse=True)[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    
    reg_model = utils.register_model(vol_size, 'nearest')
    reg_model.cuda()

    test_composed = transforms.Compose([trans.Resize_img(vol_size),
                                         trans.MinMax_norm(),
                                         trans.NumpyType((np.float32, np.int16))])
    
    test_set = datasets.OASIS3BrainDataset(test_dir, transforms=test_composed, max_dataset_size=np.inf)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    # RMSE calculation
    
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            tensor_data = [t.cuda() for t in data[:2]]
            x = tensor_data[0][0] # remove extra first dim
            y = tensor_data[1][0] # remove extra first dim
            x_patient_id = data[2][0]
            y_patient_id = data[3][0]
            x_in = torch.cat((x, y),dim=1)
            x_def, flow = model(x_in)

            flow = flow.cpu().detach().numpy()[0]
            x_def = x_def.cpu().detach().numpy()[0]
            x = x.cpu().detach().numpy()[0]
            y = y.cpu().detach().numpy()[0]
            #flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            print(flow.shape)
            print(x_def.shape, x.shape, y.shape)

            plot_result(moving=x, fixed=y, warped=x_def, outputpath='./experiments/' + model_folder, x_patient_id=x_patient_id, y_patient_id=y_patient_id)
            stdy_idx += 1

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()