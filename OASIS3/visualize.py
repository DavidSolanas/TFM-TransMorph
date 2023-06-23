import os, sys, glob
import numpy as np
import neurite as ne
import pickle
from scipy.ndimage.interpolation import map_coordinates, zoom
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph\\Baseline_registration_models')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph\\Baseline_registration_models\\VoxelMorph')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph\\OASIS')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph\\OASIS\\TransMorph')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph\\OASIS\\TransMorph\\data')

from TransMorph.Baseline_registration_models.VoxelMorph import models, utils
from TransMorph.OASIS.TransMorph.data import datasets, trans

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def main():
    data_dir = './data/OASIS_L2R_2021_task03/Submit/submission/Vxm_test/'
    ground_truth_dir = './data/OASIS_L2R_2021_task03/Test/'

    resized = (80, 96, 112) # resize img to half the original size
    test_composed = transforms.Compose([trans.MinMax_norm(), trans.Resize_img(resized), trans.NumpyType((np.float32, np.int16))])

    test_set = datasets.OASISBrainInferDataset(glob.glob(ground_truth_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    file_names = glob.glob(ground_truth_dir + '*.pkl')

    for file_pred, data_gt in zip(os.listdir(data_dir), test_loader):
        npz = np.load(data_dir + file_pred)
        x_def, flow = npz['x_def'][0], npz['flow']
        data = [t.cuda() for t in data_gt]
        x = data[0]
        y = data[1]
        x_seg = data[2]
        y_seg = data[3]
        x = x.cpu().numpy().squeeze(axis=0)[0]
        y = y.cpu().numpy().squeeze(axis=0)[0]
        print(x.shape, np.amax(x), np.amax(x_def), np.amin(x), np.amin(x_def))
        print(y.shape, np.amax(y), np.amax(x_def), np.amin(y), np.amin(x_def))

        titles = ['fixed', 'fixed', 'fixed', 'moving', 'moving', 'moving', 'deformed', 'deformed', 'deformed', 
        'diff', 'diff', 'diff', 'pre-diff', 'pre-diff', 'pre-diff']
        mid_slices_moving = [np.take(x, x.shape[d]//2, axis=d) for d in range(3)]
        mid_slices_moving[1] = np.rot90(mid_slices_moving[1], 1)
        mid_slices_moving[2] = np.rot90(mid_slices_moving[2], -1)

        mid_slices_deformed = [np.take(x_def, x_def.shape[d]//2, axis=d) for d in range(3)]
        mid_slices_deformed[1] = np.rot90(mid_slices_deformed[1], 1)
        mid_slices_deformed[2] = np.rot90(mid_slices_deformed[2], -1)
    #
        mid_slices_fixed = [np.take(y, y.shape[d]//2, axis=d) for d in range(3)]
        mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
        mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)
        
        diff_vol = y- x_def
        mid_slices_diff = [np.take(diff_vol, diff_vol.shape[d]//2, axis=d) for d in range(3)]
        mid_slices_diff[1] = np.rot90(mid_slices_diff[1], 1)
        mid_slices_diff[2] = np.rot90(mid_slices_diff[2], -1)
        
        diff_vol = y - x
        mid_slices_diffpre = [np.take(diff_vol, diff_vol.shape[d]//2, axis=d) for d in range(3)]
        mid_slices_diffpre[1] = np.rot90(mid_slices_diffpre[1], 1)
        mid_slices_diffpre[2] = np.rot90(mid_slices_diffpre[2], -1)
        
        ne.plot.slices(mid_slices_fixed + mid_slices_moving + mid_slices_deformed + mid_slices_diff + mid_slices_diffpre, 
        cmaps=['gray'], titles=titles, do_colorbars=True, grid=[5,3]);


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