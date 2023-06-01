#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.
Please make sure to use trained models appropriately. Let's say we have a model trained to register
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:
    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.h5 
        --moved moved.nii.gz --warp warp.nii.gz
The source and target input images are expected to be affinely registered.
If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 
Copyright 2020 Adrian V. Dalca
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import nibabel as nib
import neurite as ne

def plot_result(moving, fixed, warped):
    img = moving - fixed

    images = [img[0, :, :, 100,0], img[0, :, 100, :,0], img[0, 100, :, :,0]]

    # visualize
    ne.plot.slices(images, cmaps=['gray'], do_colorbars=True, titles=['Moving - fixed']*3)


    img = warped - fixed

    images = [img[0, :, :, 100,0], img[0, :, 100, :,0], img[0, 100, :, :,0]]

    # visualize
    ne.plot.slices(images, cmaps=['gray'], do_colorbars=True, titles=['Warped - fixed']*3)


def load_vols(moving_path, fixed_path):
    
    # print( "Cargando...", val_files[patient])
    
    # img = nib.load(val_files[patient]).get_fdata()
    
    img = nib.load(moving_path).get_fdata()
    
    img = img[..., np.newaxis]
    img = img[np.newaxis, ...]
    
    mm = np.max(img)
    m = np.min(img)

    moving = (img-m) / (mm-m)

    img = nib.load(fixed_path).get_fdata()
    
    img = img[..., np.newaxis]
    img = img[np.newaxis, ...]
    
    mm = np.max(img)
    m = np.min(img)

    fixed = (img-m) / (mm-m)
    
    
    return moving, fixed


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='keras model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load moving and fixed images
add_feat_axis = not args.multichannel
moving, fixed = load_vols(args.moving, args.fixed)

inshape = moving.shape[1:-1]
nb_feats = moving.shape[-1]

with tf.device(device):
    # load model and predict
    config = dict(inshape=inshape, input_model=None)
    warp = vxm.networks.VxmDense.load(args.model, **config).register(moving, fixed)
    moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

plot_result(moving, fixed, moved)