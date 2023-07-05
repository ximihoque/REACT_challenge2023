import os
import torch
from torch.utils import data
from torchvision import transforms
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
import time
import pandas as pd
from PIL import Image
import soundfile as sf
import cv2
from torch.utils.data import DataLoader
from multiprocessing import Pool
import torchaudio
from scipy.io import loadmat
torchaudio.set_audio_backend("sox_io")
from functools import cmp_to_key

random.seed(72)

class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img

def handle_noxi_emotion(video_path):
    if 'NoXI' in video_path:
        if 'Expert_video' in video_path:
            return video_path.replace('Expert_video', 'P1')
        if 'Novice_video' in video_path:
            return video_path.replace('Novice_video', 'P2')
        
    else:
        return video_path
def handle_recola_emotion(video_path):
    if 'RECOLA' in video_path:
        if 'P25' in video_path:
            return video_path.replace('P25', 'P1')
        if 'P26' in video_path:
            return video_path.replace('P26', 'P2')
        
        if 'P41' in video_path:
            return video_path.replace('P41', 'P1')
        if 'P42' in video_path:
            return video_path.replace('P42', 'P2')
        
        if 'P45' in video_path:
            return video_path.replace('P45', 'P1')
        if 'P46' in video_path:
            return video_path.replace('P46', 'P2')
    else:
        return video_path

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def extract_video_features(video_path, img_transform):
    video_list = []
    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = img_transform(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0)
        video_list.append(frame)
    video_clip = torch.cat(video_list, axis=0)
    return video_clip, fps, n_frames

def sample_audio(audio, fps, n_frames, src_sr, target_sr):
    """Explicitly for torch audio ops
    """
    audio = torchaudio.functional.resample(audio, src_sr, target_sr)
    sr = target_sr
    audio = audio.mean(0)
    frame_n_samples = int(sr / fps)
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    return audio, frame_n_samples

def load_audio(audio_path, fps, n_frames):
    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(-1)
    frame_n_samples = int(sr / fps)
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    return audio

def extract_audio_features(audio_path, fps, n_frames):
    # video_id = osp.basename(audio_path)[:-4]
    audio = load_audio(audio_path, fps, n_frames)
    shifted_n_samples = 0
    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), sample_frequency=sr, use_energy=True)
        curr_mfcc = curr_mfcc.transpose(0, 1) # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)
        curr_feat = curr_mfccs
        # rms = librosa.feature.rms(curr_samples, sr).reshape(-1)
        # zcr = librosa.feature.zero_crossing_rate(curr_samples, sr).reshape(-1)
        # curr_feat = np.concatenate((curr_mfccs, rms, zcr))

        curr_feats.append(curr_feat)

    curr_feats = np.stack(curr_feats, axis=0)
    return curr_feats


class ReactionDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, root_path, split, img_size=256, crop_size=224, clip_length=750, fps=25,
                 load_audio=True, load_video_s=True, load_video_l=True, load_emotion_s=False, load_emotion_l=False,
                 load_3dmm_s=False, load_3dmm_l=False, load_ref=True, repeat_mirrored=True, use_raw_audio=False, load_video_orig=False, mode='train'):
        """
        Args:
            root_path: (str) Path to the data folder.
            split: (str) 'train' or 'val' or 'test' split.
            img_size: (int) Size of the image.
            crop_size: (int) Size of the crop.
            clip_length: (int) Number of frames in a clip.
            fps: (int) Frame rate of the video.
            load_audio: (bool) Whether to load audio features.
            load_video_s: (bool) Whether to load speaker video features.
            load_video_l: (bool) Whether to load listener video features.
            load_emotion: (bool) Whether to load emotion labels.
            load_3dmm: (bool) Whether to load 3DMM parameters.
            repeat_mirrored: (bool) Whether to extend dataset with mirrored speaker/listener. This is used for val/test.
        """
        self._root_path = root_path
        # self._img_loader = pil_loader
        self._clip_length = clip_length
        self._fps = fps
        self.img_size = img_size
        self._split = split
        self._mode = mode
        self._data_path = self._root_path
        # print ('CLIP LEN: ', clip_length)
        # read train.csv/ val.csv
        self._list_path = pd.read_csv(split, header=None, delimiter=',')
        self._list_path = self._list_path.drop(0)

        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_video_orig = load_video_orig
        self.load_ref = load_ref
        self.use_raw_audio = use_raw_audio
    
        self._audio_path = os.path.join(self._data_path, 'Audios')
        self._video_path = os.path.join(self._data_path, 'Videos')
        self._emotion_path = os.path.join(self._data_path, 'Emotions')
        self._3dmm_path = os.path.join(self._data_path, '3D_FVs')

        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, -1)
        self.std_face = torch.FloatTensor(
            np.load('external/FaceVerse/std_face.npy').astype(np.float32)).view(1, -1)
        
        self._transform = Transform(img_size, crop_size)
        self._img_loader = pil_loader
     
        speaker_path = list(self._list_path.values[:, 1])
        listener_path = list(self._list_path.values[:, 2])
        
        if mode == 'train':
            listener_neg_path = list(self._list_path.values[:, 3])

        # TODO: FINAL training with OR
        if  repeat_mirrored: # training is always mirrored as data augmentation
            print ('mirrorring')
            speaker_path_tmp = speaker_path + listener_path
            listener_path_tmp = listener_path + speaker_path
            speaker_path = speaker_path_tmp
            listener_path = listener_path_tmp
            if mode == 'train':
                listener_path_tmp_neg = listener_neg_path + listener_neg_path
                listener_neg_path = listener_path_tmp_neg

        self.data_list = []
        if mode == 'train':
            for i, (sp, lp, lp_neg) in enumerate(zip(speaker_path, listener_path, listener_neg_path)):
                _json = self.parse_paths(sp, lp, lp_neg)
                if _json:
                    self.data_list.append(_json)
        else:
            for i, (sp, lp) in enumerate(zip(speaker_path, listener_path)):
                _json = self.parse_paths(sp, lp)
                if _json:
                    self.data_list.append(_json)
                

        self._len =  len(self.data_list)

    def parse_paths(self, sp, lp, lp_neg=None):
        ab_speaker_video_path = os.path.join(self._video_path, sp + '_marlin.pt')
        ab_speaker_video_path_dir = os.path.join(self._video_path, sp)
        if self.use_raw_audio:
            ab_speaker_audio_path = os.path.join(self._audio_path, sp +'.wav')    
        else:
            ab_speaker_audio_path = os.path.join(self._audio_path, sp +'_audio.npy')
        
        ab_speaker_emotion_path = os.path.join(self._emotion_path, sp +'.csv')
        ab_speaker_emotion_path = handle_noxi_emotion(ab_speaker_emotion_path)
        ab_speaker_emotion_path = handle_recola_emotion(ab_speaker_emotion_path)

        ab_speaker_3dmm_path = os.path.join(self._3dmm_path, sp + '.npy')

        ab_listener_video_path =  os.path.join(self._video_path, lp + '_marlin.pt')
        ab_listener_video_path_dir = os.path.join(self._video_path, lp)
        if self.use_raw_audio:
            ab_listener_audio_path = os.path.join(self._audio_path, lp +'.wav')
        else:
            ab_listener_audio_path = os.path.join(self._audio_path, lp +'_audio.npy')

        ab_listener_emotion_path = os.path.join(self._emotion_path, lp +'.csv')
        if lp_neg:
            ab_listener_emotion_path_neg = os.path.join(self._emotion_path, lp_neg +'.csv')
            ab_listener_emotion_path_neg = handle_noxi_emotion(ab_listener_emotion_path_neg)
            ab_listener_emotion_path_neg = handle_recola_emotion(ab_listener_emotion_path_neg)
            ab_listener_video_path_neg =  os.path.join(self._video_path, lp_neg + '_marlin.pt')

        ab_listener_emotion_path = handle_noxi_emotion(ab_listener_emotion_path)
        ab_listener_emotion_path = handle_recola_emotion(ab_listener_emotion_path)
        
        ab_listener_3dmm_path = os.path.join(self._3dmm_path, lp + '.npy')
        
        _json = {
                'speaker_video_path':ab_speaker_video_path, 
                'speaker_video_dir': ab_speaker_video_path_dir,
                'speaker_audio_path':ab_speaker_audio_path, 
                'speaker_emotion_path':ab_speaker_emotion_path, 
                'speaker_3dmm_path':ab_speaker_3dmm_path, 
                'listener_video_path': ab_listener_video_path, 
                'listener_video_dir': ab_listener_video_path_dir,
                'listener_audio_path':ab_listener_audio_path, 
                'listener_emotion_path':ab_listener_emotion_path,
                'listener_3dmm_path':ab_listener_3dmm_path    
        }
        if lp_neg:
            _json['listener_emotion_path_neg'] = ab_listener_emotion_path_neg
            _json['listener_video_path_neg'] = ab_listener_video_path_neg
        
        
        if os.path.exists(ab_listener_3dmm_path):
            if os.path.exists(ab_listener_emotion_path):
                if os.path.exists(ab_speaker_emotion_path):
                    if lp_neg:
                        if os.path.exists(ab_listener_emotion_path_neg):
                            return _json
                    else:
                        return _json
                else:
                    print ("Escaping: {}, Reason: {} MISSING".format(sp, ab_speaker_emotion_path))
            else:
                print ("Escaping: {}, Reason: {} MISSING".format(sp, ab_listener_emotion_path))
        else:
            print ("Escaping: {}, Reason: {} MISSING".format(sp, ab_listener_3dmm_path))
            
                
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        data = self.data_list[index]

        # ========================= Data Augmentation ==========================
        changed_sign = 0
        # if self._split == 'train.csv':  # only done at training time
        #     changed_sign = random.randint(0, 1)

        speaker_prefix = 'speaker' if changed_sign == 0 else 'listener'
        listener_prefix = 'listener' if changed_sign == 0 else 'speaker'

        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_path = data[f'{speaker_prefix}_video_path']
        listener_video_path = data[f'{listener_prefix}_video_path']
        speaker_video_dir = data[f'{speaker_prefix}_video_dir']
        listener_video_dir = data[f'{listener_prefix}_video_dir']


        speaker_video_clip = 0
        speaker_video_clip_orig = 0
        cp = 0
        if self.load_video_s:
            try:
                speaker_video_clip = torch.load(speaker_video_path)
            except Exception as err:
                print ("Exception occurred in :", speaker_video_path)
                speaker_video_clip = torch.zeros(23, 1024)

        if self.load_video_orig:
            # total_length = speaker_video_clip.shape[0]
            img_paths = os.listdir(speaker_video_dir)
            total_length = len(img_paths)
            cp = 0
            clip = []
            img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
            for img_path in img_paths:
                img = self._img_loader(os.path.join(speaker_video_dir, img_path))
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            speaker_video_clip_orig = torch.cat(clip, dim=0)
            speaker_video_clip_orig = speaker_video_clip_orig[cp:cp + self._clip_length]
        
        # listener video clip only needed for val/test
        listener_video_clip = 0
        listener_video_clip_neg = 0
        if self.load_video_l:
            # try:
            listener_video_clip = torch.load(listener_video_path)
            if self._mode == 'train':
                listener_emotion_path_neg = data[f'{listener_prefix}_video_path_neg']
                listener_video_clip_neg = torch.load(listener_emotion_path_neg)
            # except Exception as err:
                # print ("Exception occurred in :", speaker_video_path)
                # listener_video_clip = torch.zeros(23, 1024)

#             if self._mode == 'train':
#                 listener_video_clip = torch.load(listener_video_path)
#             else:
#                 img_paths = os.listdir(listener_video_dir)
#                 try:
#                     img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
#                 except Exception as err:
#                     print ('Exception img paths: ', listener_video_dir)
#                     return self.__getitem__(index+1)
# #                 img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
#                 cp = 0
#                 clip = []
#                 for img_path in img_paths:
#                     img = self._img_loader(os.path.join(listener_video_dir, img_path))
#                     img = self._transform(img)
#                     clip.append(img.unsqueeze(0))
#                 listener_video_clip = torch.cat(clip, dim=0)
#                 listener_video_clip = listener_video_clip[cp:cp + self._clip_length]

        # ========================= Load Speaker audio clip (listener audio is NEVER needed) ==========================
        listener_audio_clip, speaker_audio_clip = 0, 0
        if self.load_audio:
            speaker_audio_path = data[f'{listener_prefix}_audio_path']
            if self.use_raw_audio:
                # print ('using raw audio')
                try:
                    speaker_audio_clip, sr = torchaudio.load(speaker_audio_path)
                    speaker_audio_clip = torchaudio.functional.resample(speaker_audio_clip, sr, 16000)
                    speaker_audio_clip, frame_n_samples = sample_audio(speaker_audio_clip, fps=self._fps, src_sr=sr, 
                                                                    target_sr=16000, n_frames=self._clip_length)
                    # print ('spk org shape: ', speaker_audio_clip.shape)
                except Exception as err:
                    print ("Exception occurred in :", speaker_audio_path)
                    speaker_audio_clip = torch.zeros(480000)
                    # return self.__getitem__(index+1)

                
            else:
                try:
                    speaker_audio_clip = np.load(speaker_audio_path, allow_pickle=True)
                except Exception as err:
                    print ("Exception occurred in :", speaker_audio_path)
                    return self.__getitem__(index+1)

#               speaker_audio_clip = extract_audio_features(speaker_audio_path, self._fps, total_length)
                speaker_audio_clip = speaker_audio_clip[cp:cp + self._clip_length]


        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion, speaker_emotion = 0, 0
        cp = 0
        if self.load_emotion_l:
            
            listener_emotion_path = data[f'{listener_prefix}_emotion_path']
            listener_emotion = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
            listener_emotion = torch.from_numpy(np.array(listener_emotion.drop(0)).astype(np.float32))[
                               cp: cp + self._clip_length]
        if self._mode == 'train':
            listener_emotion_path_neg = data[f'{listener_prefix}_emotion_path_neg']
            listener_emotion_neg = pd.read_csv(listener_emotion_path_neg, header=None, delimiter=',')
            listener_emotion_neg = torch.from_numpy(np.array(listener_emotion_neg.drop(0)).astype(np.float32))[
                               cp: cp + self._clip_length]
            
        if self.load_emotion_s:
            speaker_emotion_path = data[f'{speaker_prefix}_emotion_path']
            speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')
            speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))[
                               cp: cp + self._clip_length]

        # ========================= Load Speaker & Listener 3DMM ==========================
        listener_3dmm = 0
        if self.load_3dmm_l:
            listener_3dmm_path = data[f'{listener_prefix}_3dmm_path']
            listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
            # print ('list org shape: ', listener_3dmm.shape)
            listener_3dmm = listener_3dmm[cp: cp + self._clip_length]
            listener_3dmm = listener_3dmm - self.mean_face

        speaker_3dmm = 0
        if self.load_3dmm_s:
            speaker_3dmm_path = data[f'{speaker_prefix}_3dmm_path']
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[cp: cp + self._clip_length]
            # print ('speaker 3dmm', speaker_3dmm.shape)
            # print ('mean face: ', self.mean_face.shape)
            speaker_3dmm = speaker_3dmm - self.mean_face

        # ========================= Load Listener Reference ==========================
        listener_reference = 0
        if self.load_ref:

            # listener_reference = torch.load(listener_video_path)
            # listener_reference = listener_reference.reshape(-1, 3, self.img_size, self.img_size)[0]
            listener_reference = []
            img_paths = os.listdir(listener_video_dir)
            img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
            # for img_path in img_paths:
            #         img = self._img_loader(os.path.join(speaker_video_dir, img_path))
            #         img = self._transform(img)
            #         listener_reference.append(img.unsqueeze(0))
                    # listener_reference.append(img)
            listener_reference = self._img_loader(os.path.join(listener_video_dir, img_paths[0]))
            listener_reference = self._transform(listener_reference)
            # listener_reference = torch.cat(listener_reference, dim=0)

        # print (speaker_video_clip.shape)
        # print (speaker_audio_clip.shape) 
        # print (speaker_emotion.shape) 
        # print (speaker_3dmm.shape)
        # print (listener_video_clip.shape) 
        # print (listener_audio_clip.shape) 
        # print ('list emo', listener_emotion.shape) 
        # print ('list 3d', listener_3dmm.shape) 
        # print (listener_reference.shape)
        if self._mode == 'train':
            return speaker_video_clip, speaker_video_clip_orig, speaker_audio_clip, speaker_emotion, speaker_3dmm, listener_video_clip, listener_video_clip_neg, listener_audio_clip, listener_emotion, listener_emotion_neg, listener_3dmm
        else:
            return speaker_video_clip, speaker_video_clip_orig, speaker_audio_clip, speaker_emotion, speaker_3dmm, listener_video_clip, listener_audio_clip, listener_emotion, listener_3dmm, listener_reference

    def __len__(self):
        return self._len


def get_dataloader(conf, split, load_audio=False, load_video_s=False, load_video_l=False, load_emotion_s=False,
                   load_emotion_l=False, load_3dmm_s=False, load_3dmm_l=False, load_ref=False, repeat_mirrored=False, 
                   use_raw_audio=False, load_video_orig=False, mode='train'):
    
#     assert split in ["train", "val", "test"], "split must be in [train, val, test]"
    #print('==> Preparing data for {}...'.format(split) + '\n')
    print (
        'MODE: ', mode
    )
    dataset = ReactionDataset(conf.dataset_path, split, 
                              img_size=conf.img_size, 
                              crop_size=conf.crop_size,
                              clip_length=conf.seq_len,
                              load_audio=load_audio, 
                              load_video_s=load_video_s, 
                              load_video_l=load_video_l,
                              load_emotion_s=load_emotion_s, 
                              load_emotion_l=load_emotion_l, 
                              load_3dmm_s=load_3dmm_s,
                              load_3dmm_l=load_3dmm_l, 
                              load_ref=load_ref, 
                              load_video_orig=load_video_orig,
                              repeat_mirrored=True,
                              use_raw_audio=use_raw_audio,
                              mode=mode)
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    dataloader = DataLoader(dataset=dataset, batch_size=conf.batch_size, shuffle=shuffle, num_workers=conf.num_workers)
    return dataloader
