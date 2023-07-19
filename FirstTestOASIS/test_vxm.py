import glob
import os, sys
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
from torchvision import transforms
from natsort import natsorted
from scipy.ndimage.interpolation import map_coordinates, zoom

sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph\\Baseline_registration_models')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph\\Baseline_registration_models\\VoxelMorph')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph\\OASIS')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph\\OASIS\\TransMorph')
sys.path.append('C:\\Users\\david\\Documents\\Master Ingenieria Informatica\\TFM\\code\\TransMorph\\OASIS\\TransMorph\\data')

from TransMorph.Baseline_registration_models.VoxelMorph import models, utils
from TransMorph.OASIS.TransMorph.data import datasets, trans
import TransMorph.OASIS.TransMorph.utils as OASIS_utils


def main():
    test_dir ='./data/OASIS_L2R_2021_task03/Test/'
    save_dir = './data/OASIS_L2R_2021_task03/Submit/submission/Vxm_test/'
    model_idx = -1
    img_size_orig = (160, 192, 224)
    resized = (80, 96, 112) # resize img to half the original size
    weights = [1, 0.02]
    model_folder = 'OASIS_test_MI/'
    model_dir = 'experiments/' + model_folder
    
    model = models.VxmDense_1(resized)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = OASIS_utils.register_model(resized, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Resize_img(resized), #rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.float32))])
    test_set = datasets.OASISBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    file_names = glob.glob(test_dir + '*.pkl')

    # RMSE calculation
    criterion = nn.MSELoss()
    loss_all = utils.AverageMeter()
    rmse = 0
    dice_all = utils.AverageMeter()
    
    with torch.no_grad():
        stdy_idx = 0
        for data, file_name in zip(test_loader, file_names):
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            x_in = torch.cat((x, y),dim=1)
            x_def, flow = model(x_in)
            rmse += torch.sqrt(criterion(x_def, y))
            loss_all.update(rmse.item(), y.numel())

            flow = flow.cpu().detach().numpy()[0]
            x_def = x_def.cpu().detach().numpy()[0]
            flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            print(flow.shape)
            print(x_def.shape)
            file_name = file_names[stdy_idx].split('\\')[-1].split('.')[0][2:]
            np.savez(save_dir+'disp_{}.npz'.format(file_name), x_def=x_def, flow=flow)
            stdy_idx += 1
        print('RMSE: ', loss_all.avg)

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