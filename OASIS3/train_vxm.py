# imports
import os, sys
import glob

# third party imports
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter

# os.getcwd() is assumed to be in a folder inside the root git repo folder
code_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(code_folder)
sys.path.append(code_folder + '/Baseline_registration_models')
sys.path.append(code_folder + '/Baseline_registration_models/VoxelMorph')
sys.path.append(code_folder + '/OASIS')
sys.path.append(code_folder + '/OASIS/TransMorph')
sys.path.append(code_folder + '/OASIS/TransMorph/data')


from data import trans, datasets
from compute_dice import dice
from Baseline_registration_models.VoxelMorph import utils, losses, models
from OASIS.TransMorph import utils as utils_OASIS
from Baseline_registration_models.VoxelMorph.train_vxm import adjust_learning_rate, mk_grid_img, comput_fig, Logger, save_checkpoint


def main():
    ROOT_DATA_DIR = '/Sauron1/david/tfm/dataset/OASIS3_final/'
    ROOT_LOG_DIR = '/Sauron1/david/tfm/tfm-tests/oasis3_final_pairs/'
    batch_size = 1
    max_dataset_size = 500
    train_dir = ROOT_DATA_DIR + 'train/'
    val_dir = ROOT_DATA_DIR + 'test/'
    save_dir = 'OASIS3_final_pairs_MSE_Interpn_train_val/'
    #save_dir = 'test/'

    if not os.path.exists(ROOT_LOG_DIR + 'experiments/' + save_dir):
        os.makedirs(ROOT_LOG_DIR + 'experiments/' + save_dir)
    if not os.path.exists(ROOT_LOG_DIR + 'logs/' + save_dir):
        os.makedirs(ROOT_LOG_DIR + 'logs/' + save_dir)
    sys.stdout = Logger(ROOT_LOG_DIR + 'logs/' + save_dir)

    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 50 #max traning epoch
    vol_size = (128, 128, 128) # resize img to half the original size
    weights = [1, 0.02]
    cont_training = True

    '''
    Initialize model
    '''

    model = models.VxmDense_2(vol_size)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(vol_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(vol_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 29
        model_idx = -1
        model_dir = ROOT_LOG_DIR + 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
        model.load_state_dict(best_model)
        print('Best model: {}'.format(glob.glob(model_dir + natsorted(os.listdir(model_dir))[model_idx])))
    else:
        updated_lr = lr


    '''
    Initialize training
    '''
    vxm_transforms = transforms.Compose([trans.OASIS3Crop(),
                                     trans.Resize_img(vol_size),
                                     trans.MinMax_norm(),
                                     trans.NumpyType((np.float32, np.int16))])


    # TODO
    train_set = datasets.OASIS3BrainDataset(train_dir, transforms=vxm_transforms, max_dataset_size=max_dataset_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_set = datasets.OASISBrainInferDataset(val_dir, transforms=vxm_transforms, max_dataset_size=max_dataset_size)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)


    # prepare losses and compile
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = nn.MSELoss()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    writer = SummaryWriter(log_dir=ROOT_LOG_DIR + 'logs/'+save_dir)
    
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0][0] # remove extra first dim
            y = data[1][0] # remove extra first dim
            x_in = torch.cat((x,y),dim=1)
            output = model(x_in)   
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch {}: Iter {} of {} loss {:.6f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(epoch, idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))
        
        writer.add_scalar('Loss/train', loss_all.avg, epoch)

        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, vol_size)
                output = model(x_in)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                dsc = dice(def_out.long(), y_seg.long(), nlabels=32)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=ROOT_LOG_DIR + 'experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        
        # Save always last model
        if epoch == max_epoch - 1:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            }, save_dir=ROOT_LOG_DIR + 'experiments/' + save_dir, filename='dsc{:.3f}_latest.pth.tar'.format(eval_dsc.avg),
            max_model_num=999)

        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        

        loss_all.reset()
    writer.close()
        

    

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
