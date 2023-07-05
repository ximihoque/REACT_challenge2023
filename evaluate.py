import os
import numpy as np
import torch
# torch.set_num_threads(1)

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
import logging
from model import TransformerVAEEmtMarlin
from utils import AverageMeter
from render import Render
from model.losses import VAELoss
from metric import *
from dataset import get_dataloader
from utils import load_config
import model as module_arch
import model.losses as module_loss
from functools import partial

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--dataset-path', default="../data", type=str, help="dataset path")
    parser.add_argument('--split', type=str, help="split of dataset", choices=["val", "test"], required=True)
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('--max-seq-len', default=751, type=int, help="max length of clip")
    parser.add_argument('--seq-len', default=750, type=int, help="length of clip")
    parser.add_argument('--clip-length', default=751, type=int, help="len of video clip")
    parser.add_argument('--window-size', default=8, type=int, help="prediction window-size for online mode")
    parser.add_argument('--feature-dim', default=128, type=int, help="feature dim of model")
    parser.add_argument('--audio-dim', default=78, type=int, help="feature dim of audio")
    parser.add_argument('--_3dmm-dim', default=58, type=int, help="feature dim of 3dmm")
    parser.add_argument('--emotion-dim', default=25, type=int, help="feature dim of emotion")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.0002, type=float, help="hyperparameter for kl-loss")
    parser.add_argument('--threads', default=16, type=int, help="num max of threads")
    parser.add_argument('--binarize', action='store_true', help='binarize AUs output from model')
    parser.add_argument('--use-video',  default=False, action='store_true', help='w/ or w/o video modality')

    args = parser.parse_args()
    return args

# Evaluating
def val(args, model, val_loader, criterion, render, binarize=False):
    losses = AverageMeter()
    rec_losses = AverageMeter()
    kld_losses = AverageMeter()

    out_dir = os.path.join(args.outdir, args.split)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    listener_emotion_gt_list = []
    listener_emotion_pred_list = []
    speaker_emotion_list = []
    all_listener_emotion_pred_list = []

    for batch_idx, (speaker_video, speaker_video_clip_orig, _, speaker_emotion, _, _, _, listener_emotion, listener_3dmm, listener_references) in enumerate(tqdm(val_loader)):
        if torch.cuda.is_available():
            speaker_emotion,  listener_emotion, listener_3dmm = \
                speaker_emotion.cuda(), listener_emotion.cuda(), listener_3dmm.cuda()

            if args.use_video:
                speaker_video = speaker_video.cuda()
                
            else:
                speaker_video = None
        with torch.no_grad():
            prediction = model(speaker_emotion=speaker_emotion, speaker_video=speaker_video, is_train=False)
        
            listener_3dmm_out, listener_emotion_out, distribution = prediction
            loss, rec_loss, kld_loss = criterion(listener_emotion, listener_3dmm, listener_emotion_out, listener_3dmm_out, distribution)

            losses.update(loss.data.item(), speaker_emotion.size(0))
            rec_losses.update(rec_loss.data.item(), speaker_emotion.size(0))
            kld_losses.update(kld_loss.data.item(), speaker_emotion.size(0))
        
            # binarize first 15 positions
            if binarize:
                listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])
           
            listener_emotion_pred_list.append(listener_emotion_out.cpu())
            listener_emotion_gt_list.append(listener_emotion.cpu())
            speaker_emotion_list.append(speaker_emotion.cpu())

    listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim = 0)
    listener_emotion_gt = torch.cat(listener_emotion_gt_list, dim = 0)
    speaker_emotion_gt = torch.cat(speaker_emotion_list, dim = 0)
    all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))

    print("-----------------Repeat 9 times-----------------")
    for i in range(9):
        listener_emotion_pred_list = []
        for batch_idx, (speaker_video, speaker_video_clip_orig, _, speaker_emotion, _, _, _, listener_emotion, listener_3dmm, listener_references) in enumerate(tqdm(val_loader)):
            if torch.cuda.is_available():
                speaker_emotion,  listener_emotion, listener_3dmm = \
                    speaker_emotion.cuda(), listener_emotion.cuda(), listener_3dmm.cuda()

                if args.use_video:
                    speaker_video = speaker_video.cuda()
                    
                else:
                    speaker_video = None
            with torch.no_grad():
                prediction = model(speaker_emotion=speaker_emotion, speaker_video=speaker_video, is_train=False)
                listener_emotion_out = prediction[1]
              
                # binarize first 15 positions
                if binarize:
                    listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])
                listener_emotion_pred_list.append(listener_emotion_out.cpu())
        listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim=0)
        all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))
        
    all_listener_emotion_pred = torch.cat(all_listener_emotion_pred_list, dim=1)

    print("-----------------Evaluating Metric-----------------")
    torch.cuda.empty_cache()
    p = args.threads
    
    print ("computing TLCC")
    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    TLCC = compute_TLCC_mp(all_listener_emotion_pred, speaker_emotion_gt, p)
    
    print ("computing FRC")
    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    FRC = compute_FRC_mp(args, all_listener_emotion_pred, listener_emotion_gt, val_test=args.split, p=p)

    print("computing FRD")
    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD_mp(args, all_listener_emotion_pred, listener_emotion_gt, val_test=args.split, p=p)

    print ("computing rest")
    FRDvs = compute_FRDvs(all_listener_emotion_pred)
    FRVar  = compute_FRVar(all_listener_emotion_pred)
    smse  = compute_s_mse(all_listener_emotion_pred)

    return losses.avg, rec_losses.avg, kld_losses.avg, FRC, FRD, FRDvs, FRVar, smse, TLCC



def main(args):
    checkpoint_path = args.resume
   
    if args.split == 'val':
        split = '../data/val.csv'
    else:
        split = '../data/test.csv'

    val_loader = get_dataloader(args, split, load_audio=False, load_video_s=True, load_emotion_s=True, load_emotion_l=True, load_3dmm_l=True, load_ref=False, load_video_orig=False, mode='val')
    model = TransformerVAEEmtMarlin(output_emotion_dim = args.emotion_dim, output_3dmm_dim = args._3dmm_dim, feature_dim = args.feature_dim, seq_len = args.seq_len, online = False,  device = args.device)
    criterion = VAELoss(args.kl_p)
    

    if args.resume != '': #  resume from a checkpoint
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        print ('GPUUUU')
        model = model.cuda()
        # render = Render('cuda')
    # else:
    # render = Render()

    val_loss, rec_loss, kld_loss, FRC, FRD, FRDvs, FRVar, smse, TLCC = val(args, model, val_loader, criterion, 0, binarize=args.binarize)
    print("{}_loss: {:.5f}   {}_rec_loss: {:.5f}  {}_kld_loss: {:.5f} ".format(args.split, val_loss, args.split, rec_loss, args.split, kld_loss))
    print("Metric: | FRC: {:.5f} | FRD: {:.5f} | S-MSE: {:.5f} | FRVar: {:.5f} | FRDvs: {:.5f} | TLCC: {:.5f}".format(FRC, FRD, smse, FRVar, FRDvs, TLCC))
    print("Latex-friendly --> model_name & {:.2f} & {:.2f} & {:.4f} & {:.4f} & {:.4f} & - & {:.2f} \\\\".format( FRC, FRD, smse, FRVar, FRDvs, TLCC))


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    args = parse_arg()
    torch.multiprocessing.set_start_method('spawn')

    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args)

