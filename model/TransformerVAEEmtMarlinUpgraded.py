import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .BasicBlock import ConvBlock, PositionalEncoding, lengths_to_mask, init_biased_mask
# from .HuBert import HuBERTEncoder
# from train_siamese import SiameseNetwork

class VAEModel(nn.Module):
    def __init__(self,
                 latent_dim: int = 256,
                 device='cuda',
                 **kwargs) -> None:
        super(VAEModel, self).__init__()

        self.latent_dim = latent_dim
        self.device = device 
        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=latent_dim * 2,
                                                             dropout=0.3)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=1)
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))


    def forward(self, x):
        B, T, D = x.shape
        lengths = [len(item) for item in x]

        mu_token = torch.tile(self.mu_token, (B,)).reshape(B, 1, -1)
        logvar_token = torch.tile(self.logvar_token, (B,)).reshape(B, 1, -1)

        x = torch.cat([mu_token, logvar_token, x], dim=1)

        x = x.permute(1, 0, 2)

        token_mask = torch.ones((B, 2), dtype=bool, device=self.device)
        mask = lengths_to_mask(lengths, device=self.device)
        aug_mask = torch.cat((token_mask, mask), 1)

        x = self.seqTransEncoder(x, src_key_padding_mask=~aug_mask)

        mu = x[0]
        logvar = x[1]
        std = logvar.exp().pow(0.5)
        # print ('mu', mu.shape)
        # print ('logvar', logvar.shape)
        dist = torch.distributions.Normal(mu, std)
        # print ('dist', dist)
        motion_sample = self.sample_from_distribution(dist).to(self.device)

        return motion_sample, dist

    def sample_from_distribution(self, distribution):
         return distribution.rsample()


class Decoder(nn.Module):
    def __init__(self,  output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, device = 'cpu', 
                        max_seq_len=751, n_head = 4, window_size = 8, online = False):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.window_size = window_size
        self.device = device
        
        self.vae_model = VAEModel(feature_dim, device=device)

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)
        self.listener_reaction_decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.listener_reaction_decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)


        self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = max_seq_len, period=max_seq_len)

        self.listener_reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)

        #TODO: create two emotion map layers, 1 - AU (Sigmoid activation), 2 - Emotions 
        self.listener_reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim + output_3dmm_dim, feature_dim),
            nn.Linear(feature_dim, output_emotion_dim)
        )
        # self.listener_emotion_AU = nn.Sequential(
        #     nn.Linear(output_emotion_dim, 15),
        #     nn.ReLU(),
        #     nn.Sigmoid()
        # )
        # self.listener_emotion_emotion = nn.Linear(output_emotion_dim, 10)

        self.PE = PositionalEncoding(feature_dim)


    def forward(self, encoded_feature):
        B = encoded_feature.shape[0]
        TL = 750
        motion_sample, dist = self.vae_model(encoded_feature)

        time_queries = torch.zeros(B, TL, self.feature_dim, device=encoded_feature.get_device())
        time_queries = self.PE(time_queries)
        tgt_mask = self.biased_mask[:, :TL, :TL].clone().detach().to(device=self.device).repeat(B,1,1)

        listener_reaction = self.listener_reaction_decoder_1(tgt=time_queries, memory=motion_sample.unsqueeze(1), tgt_mask=tgt_mask)
        listener_reaction = self.listener_reaction_decoder_2(listener_reaction, listener_reaction, tgt_mask=tgt_mask)

       
        listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)


        listener_emotion_out = self.listener_reaction_emotion_map_layer(
            torch.cat((listener_3dmm_out, listener_reaction), dim=-1))

        return listener_3dmm_out, listener_emotion_out, dist

    def reset_window_size(self, window_size):
        self.window_size = window_size

class VideoEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(VideoEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=feature_dim, kernel_size=3)
       
        self.pool = nn.AdaptiveMaxPool1d(23)
        self.relu = nn.ReLU()
        transformer_layer = nn.TransformerEncoderLayer(d_model=feature_dim, 
                                                    nhead=4,
                                                    dim_feedforward=feature_dim*2,
                                                    dropout=0.3)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

    def forward(self, x):
        out = x.permute(0, 2, 1) # N x features x seq
        out = self.relu(self.conv1(out))
        out = self.pool(out)
        out = out.permute(0, 2, 1) # N x seq x features
        out = self.encoder(out)
        return out

class EmotionEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(EmotionEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=25, out_channels=feature_dim, kernel_size=3)

        self.pool = nn.AdaptiveMaxPool1d(750)
        self.relu = nn.ReLU()
        transformer_layer = nn.TransformerEncoderLayer(d_model=feature_dim, 
                                                    nhead=4,
                                                    dim_feedforward=feature_dim*2,
                                                    dropout=0.3)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

    def forward(self, x):
        out = x.permute(0, 2, 1) # N x features x seq
        out = self.pool(self.relu(self.conv1(x.permute(0, 2, 1)))) 
        out = out.permute(0, 2, 1)
        out = self.encoder(out)
        return out


class SpeakerBehaviourEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(SpeakerBehaviourEncoder, self).__init__()
        
        
        self.video_encoder = VideoEncoder(feature_dim=128)
        
        self.emotion_encoder = EmotionEncoder()
        transformer_layer = nn.TransformerEncoderLayer(d_model=128, 
                                                    nhead=4,
                                                    dim_feedforward=256,
                                                    dropout=0.3)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)
        # self.pool = nn.AdaptiveMaxPool1d(seq_len)
        self.fusion_layer =  nn.Linear(128, feature_dim)

    def forward(self, video, emotion):
        # handle if audio not present
        emotion_feature = self.emotion_encoder(emotion)
        video_feature = self.video_encoder(video)
        # print ('video: ', video_feature.shape)
        encoded_feature = self.fusion_layer(self.encoder(torch.cat((emotion_feature, video_feature), dim=1)))
        
        return encoded_feature



class TransformerVAEEmtMarlinUpgraded(nn.Module):
    def __init__(self, img_size=224, audio_dim = 78, output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, seq_len=751, max_seq_len=751, online = False, window_size = 8, use_hubert=False, device = 'cuda'):
        super(TransformerVAEEmtMarlinUpgraded, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim
        self.seq_len = seq_len
        self.online = online
        self.window_size = window_size
        self.use_hubert = use_hubert

        self.speaker_behaviour_encoder = SpeakerBehaviourEncoder(feature_dim)
     
        self.reaction_decoder = Decoder(output_3dmm_dim = output_3dmm_dim, output_emotion_dim = output_emotion_dim, feature_dim = feature_dim, max_seq_len=max_seq_len, device=device, window_size = self.window_size, online = online)
        # self.listener_processor = nn.Conv1d(in_channels=25, out_channels=feature_dim, kernel_size=3)
        # self.pool = nn.AdaptiveMaxPool1d(seq_len)

    def forward(self, speaker_video, speaker_emotion, listener_emotion=None):

        """
        input:
        video: (batch_size, seq_len, 3, img_size, img_size)
        audio: (batch_size, raw_wav)
        listener_emotion: (pos_emt, neg_emt)

        output:
        3dmm_vector: (batch_size, seq_len, output_3dmm_dim)
        emotion_vector: (batch_size, seq_len, output_emotion_dim)
        distribution: [dist_1,...,dist_n]
        """

        distribution = []
        relu = nn.ReLU()
        if listener_emotion:
            listener_emt_pos, listener_emt_neg = listener_emotion
            # print ('neg shape: ', listener_emt_neg.shape)
            # print ('pos shape: ', listener_emt_pos.shape)
            listener_emt_pos = self.speaker_behaviour_encoder.emotion_encoder(listener_emt_pos)
            listener_emt_neg = self.speaker_behaviour_encoder.emotion_encoder(listener_emt_neg)
        
        # print ('neg shape: ', listener_emt_neg.shape)
        # print ('pos shape: ', listener_emt_pos.shape)
        encoded_feature = self.speaker_behaviour_encoder(speaker_video, speaker_emotion)
        # encoded_feature = self.emotion_encoder(speaker_emotion)
        listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature)
        distribution.append(dist)

        if listener_emotion:
            return listener_3dmm_out, listener_emotion_out, distribution, encoded_feature, listener_emt_pos, listener_emt_neg
        else:
            return listener_3dmm_out, listener_emotion_out, distribution


    def reset_window_size(self, window_size):
        self.window_size = window_size
        self.reaction_decoder.reset_window_size(window_size)



if __name__ == "__main__":
    pass
