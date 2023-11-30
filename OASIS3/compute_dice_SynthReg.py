import glob
import os, sys
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
from torchvision import transforms
from natsort import natsorted
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib

# os.getcwd() is assumed to be in a folder inside the root git repo folder
code_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(code_folder)
sys.path.append(code_folder + '/Baseline_registration_models')
sys.path.append(code_folder + '/Baseline_registration_models/VoxelMorph')
sys.path.append(code_folder + '/OASIS')
sys.path.append(code_folder + '/OASIS/TransMorph')
sys.path.append(code_folder + '/OASIS/TransMorph/data')

from data import trans, datasets
from Baseline_registration_models.VoxelMorph import utils


def dice_mhg(y_pred, y_true, nlabels):
    # Convert inputs to torch tensors
    pred = torch.squeeze(y_pred)
    true = torch.squeeze(y_true)
    
    DSCs = torch.zeros((nlabels, 1))
    
    for i in range(1, nlabels+1):
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = torch.sum(intersection)
        union = torch.sum(pred_i) + torch.sum(true_i)
        dsc = (2. * intersection) / (union + 1e-5)
        DSCs[i-1] = dsc
      
    return DSCs.mean()


def dice(y_pred, y_true, nlabels=32):
    # DSC and Jaccard computation

    # Compute DICE for every region
    interVol = torch.zeros(nlabels)
    refVol = torch.zeros(nlabels)
    segVol = torch.zeros(nlabels)
    dsc = torch.zeros(nlabels)

    structs = [
        #(0, 'background'),
        (2, 'left cerebral white matter'),
        (3, 'left cerebral cortex'),
        (4, 'left lateral ventricle'),
        (5, 'left inferior lateral ventricle'),
        (7, 'left cerebellum white matter'),
        (8, 'left cerebellum cortex'),
        (10, 'left thalamus'),
        (11, 'left caudate'),
        (12, 'left putamen'),
        (13, 'left pallidum'),
        (14, '3rd ventricle'),
        (15, '4th ventricle'),
        (16, 'brain-stem'),
        (17, 'left hippocampus'),
        (18, 'left amygdala'),
        (24, 'cerebral spinal fluid'),
        (26, 'left accumbens area'),
        (28, 'left ventral DC'),
        (41, 'right cerebral white matter'),
        (42, 'right cerebral cortex'),
        (43, 'right lateral ventricle'),
        (44, 'right inferior lateral ventricle'),
        (46, 'right cerebellum white matter'),
        (47, 'right cerebellum cortex'),
        (49, 'right thalamus'),
        (50, 'right caudate'),
        (51, 'right putamen'),
        (52, 'right pallidum'),
        (53, 'right hippocampus'),
        (54, 'right amygdala'),
        (58, 'right accumbens area'),
        (60, 'right ventral DC')
    ]

    for ll in range(len(structs)):
        label, _ = structs[ll]
        
        aux = torch.sum(y_pred[y_true == label] == label)
        interVol[ll] = aux
        aux = torch.sum(y_true == label)
        refVol[ll] = aux
        aux = torch.sum(y_pred == label)
        segVol[ll] = aux

        dsc[ll] = 2.0 * interVol[ll] / (refVol[ll] + segVol[ll])
    
    return dsc.mean()


def load_volume(path):
    vol = nib.load(path).get_fdata()

    vol = np.ascontiguousarray(vol)
    vol = torch.from_numpy(vol)
    # Reshape the volume and add extra dimension
    vol = vol.permute(3, 0, 1, 2).unsqueeze(0)
    return vol


def compute_dsc_from_fixed_volumes(model: nn.Module, reg_model: nn.Module):
    model.eval()
    x_seg = load_volume('./volumes/moving_seg.nii.gz')
    y_seg = load_volume('./volumes/fixed_seg.nii.gz')
    warp = load_volume('./volumes/warp.nii.gz')

    def_out = reg_model([x_seg.cuda().float(), warp.cuda().float()])
    dsc = dice(def_out.long(), y_seg.long(), 32)
    print(dsc)



def main():
    ROOT_DATA_DIR = '/Sauron1/david/tfm/dataset/OASIS3_final/'
    
    test_dir = ROOT_DATA_DIR + 'test/'
    vol_size = (128, 128, 128) # resize img to half the original size

    #if not os.path.exists(ROOT_LOG_DIR + 'logs/' + model_folder):
    #    os.makedirs(ROOT_LOG_DIR + 'logs/' + model_folder)
    #sys.stdout = Logger(ROOT_LOG_DIR + 'logs/' + model_folder)
    

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(vol_size, 'nearest')
    reg_model.cuda()


    '''
    Initialize data
    '''
    vxm_transforms = transforms.Compose([trans.OASIS3Crop(),
                                     trans.Resize_img(vol_size),
                                     trans.MinMax_norm(),
                                     trans.NumpyType((np.float32, np.int16))])


    # TODO

    test_set = datasets.OASISBrainInferFromSynthReg(test_dir, transforms=vxm_transforms, max_dataset_size=np.inf, test=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    #writer = SummaryWriter(log_dir='logs/'+model_folder)
    
    #compute_dsc_from_fixed_volumes(model, reg_model)
    #return
    
    '''
    Validation
    '''
    eval_dsc = utils.AverageMeter()
    with torch.no_grad():
        for data in test_loader:

            data = [t.cuda() for t in data]
            x_seg = data[0]
            y_seg = data[1]
            warp = data[2]

            def_out = reg_model([x_seg.cuda().float(), warp])
            #save_nifti_volume(x, 'moving.nii.gz')
            #save_nifti_volume(y, 'fixed.nii.gz')
            #save_nifti_volume(x_seg, 'moving_seg.nii.gz')
            #save_nifti_volume(y_seg, 'fixed_seg.nii.gz')
            #save_nifti_volume(output[0].cuda(), 'moved_fakeT1.nii.gz')
            #save_nifti_volume(warp, 'warp_fakeT1.nii.gz')
            #save_nifti_volume(def_out.long().to(torch.int32), 'out_seg_warp_fakeT1.nii.gz')
            dsc = dice(def_out.long(), y_seg.long(), 32)
            eval_dsc.update(dsc.item(), x_seg.size(0))
            print(eval_dsc.avg)
            print()

    #writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
    print(f'Mean DSC: {eval_dsc.avg:.3f} ({eval_dsc.std:.3f})')



if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
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