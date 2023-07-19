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
from Baseline_registration_models.VoxelMorph import utils, models
from Baseline_registration_models.VoxelMorph.train_vxm import  mk_grid_img, comput_fig


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
    ROOT_DATA_DIR = '/Disco2021-I/david/tfm/dataset/OASIS3_processed/'
    ROOT_LOG_DIR = '/Disco2021-I/david/tfm/tfm-tests/oasis3_500_pairs/'
    
    test_dir = ROOT_DATA_DIR + 'test/'
    model_folder = 'OASIS3_500_pairs_MSE_Interpn/'
    model_idx = 0
    vol_size = (128, 128, 128) # resize img to half the original size
    model_dir = ROOT_LOG_DIR + 'experiments/' + model_folder

    if not os.path.exists(ROOT_LOG_DIR + 'logs/' + model_folder):
        os.makedirs(ROOT_LOG_DIR + 'logs/' + model_folder)
    #sys.stdout = Logger(ROOT_LOG_DIR + 'logs/' + model_folder)
    
    '''
    Load model
    '''
    model = models.VxmDense_2(vol_size)
    best_model = torch.load(glob.glob(model_dir + "/*latest*")[model_idx])['state_dict']
    print('Best model: {}'.format(glob.glob(model_dir + "/*latest*")[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(vol_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(vol_size, 'bilinear')
    reg_model_bilin.cuda()


    '''
    Initialize training
    '''
    test_composed = transforms.Compose([trans.Resize_img(vol_size),
                                         trans.MinMax_norm(),
                                         trans.NumpyType((np.float32, np.int16))])
    
    seg_composed = transforms.Compose([
        trans.Resize_img(vol_size, force_nn=True)
        ])
    
    test_set = datasets.OASISBrainInferDataset(test_dir, transforms=test_composed, max_dataset_size=np.inf, seg_transforms=seg_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    best_dsc = 0
    best_dsc_mhg = 0
    #writer = SummaryWriter(log_dir='logs/'+model_folder)
    
    #compute_dsc_from_fixed_volumes(model, reg_model)
    #return
    
    '''
    Validation
    '''
    eval_dsc = utils.AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0][0]
            y = data[1][0]
            x_seg = data[2][0]
            y_seg = data[3][0]
            x_in = torch.cat((x, y), dim=1)
            grid_img = mk_grid_img(8, 1, vol_size)
            output = model(x_in)
            def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
            def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
            #print(x.shape)            
            #print(y.shape)
            #print(x_seg.shape)
            #print(y_seg.shape)
            #print(output[0].cuda().shape)
            #print(output[1].cuda().shape)
            #print(def_out.long().shape)
            #save_nifti_volume(x, 'moving.nii.gz')
            #save_nifti_volume(y, 'fixed.nii.gz')
            #save_nifti_volume(x_seg, 'moving_seg.nii.gz')
            #save_nifti_volume(y_seg, 'fixed_seg.nii.gz')
            #save_nifti_volume(output[0].cuda(), 'moved.nii.gz')
            #save_nifti_volume(output[1].cuda(), 'warp.nii.gz')
            #save_nifti_volume(def_out.long().to(torch.int32), 'out_seg_warp.nii.gz')
            dsc = dice(def_out.long(), y_seg.long(), 32)
            eval_dsc.update(dsc.item(), x.size(0))
            print(eval_dsc.avg)
            print()
    best_dsc = max(eval_dsc.avg, best_dsc)

    #writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
    print(f'Mean DSC: {best_dsc}')
    # MSE: Mean DSC: 0.3214651603996754; 
    return
    # Create a figure with multiple subplots
    plt.switch_backend('agg')
    pred_fig = comput_fig(def_out)
    grid_fig = comput_fig(def_grid)
    x_fig = comput_fig(x_seg)
    tar_fig = comput_fig(y_seg)
    # Create a new combined figure with multiple subplots
    fig_combined = plt.figure(figsize=(12, 8))
    ax1 = fig_combined.add_subplot(221)
    ax2 = fig_combined.add_subplot(222)
    ax3 = fig_combined.add_subplot(223)
    ax4 = fig_combined.add_subplot(224)
    ax1.set_title('Grid')
    ax1.set_title('input')
    ax1.set_title('ground truth')
    ax1.set_title('prediction')
    ax1.imshow(grid_fig)
    ax2.imshow(x_fig)
    ax3.imshow(tar_fig)
    ax4.imshow(pred_fig)
    plt.tight_layout()
    plt.show()
    #writer.add_figure('Grid', grid_fig, epoch)
    plt.close(grid_fig)
    #writer.add_figure('input', x_fig, epoch)
    plt.close(x_fig)
    #writer.add_figure('ground truth', tar_fig, epoch)
    plt.close(tar_fig)
    #writer.add_figure('prediction', pred_fig, epoch)
    plt.close(pred_fig)
    #writer.close()


def save_nifti_volume(data: torch, outpath: str):
    # Assuming you have a Torch tensor called 'data' with shape (C, D, H, W)
    # C: number of channels, D: depth, H: height, W: width

    # Convert the Torch tensor to a NumPy array
    data_np = data[0].cpu().numpy()

    # Transpose the array to match the NIfTI format (D, H, W, C)
    data_np = np.transpose(data_np, (1, 2, 3, 0))

    # Create a NIfTI image object
    nifti_img = nib.Nifti1Image(data_np, affine=None)  # Set the 'affine' parameter if needed

    # Save the NIfTI image to a file
    nib.save(nifti_img, outpath)

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


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