# Submission for REACT Challenge 2020

### Challenge Description
Human behavioural responses are stimulated by their environment (or context), and people will inductively process the stimulus and modify their interactions to produce an appropriate response. When facing the same stimulus, different facial reactions could be triggered across not only different subjects but also the same subjects under different contexts. The Multimodal Multiple Appropriate Facial Reaction Generation Challenge (REACT 2023) is a satellite event of ACM MM 2023, (Ottawa, Canada, October 2023), which aims at comparison of multimedia processing and machine learning methods for automatic human facial reaction generation under different dyadic interaction scenarios. The goal of the Challenge is to provide the first benchmark test set for multimodal information processing and to bring together the audio, visual and audio-visual affective computing communities, to compare the relative merits of the approaches to automatic appropriate facial reaction generation under well-defined conditions.


#### Task 1 - Offline Appropriate Facial Reaction Generation
This task aims to develop a machine learning model that takes the entire speaker behaviour sequence as the input, and generates multiple appropriate and realistic / naturalistic spatio-temporal facial reactions, consisting of AUs, facial expressions, valence and arousal state representing the predicted facial reaction. As a result,  facial reactions are required to be generated for the task given each input speaker behaviour.

## Coding environment and Dependencies

- Python 3.8+ 
- PyTorch 1.9+
- CUDA 11.1+ 


```shell
conda create -n react python=3.8
conda activate react
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
## Input and Output of the model.
 - Input : MARLIN Features of Videos data + Emotions data provided in the challenge dataset
 - Output : 3DMM coefficients(52 facial expression coefficients, 3 pose coefficients and 3 translation coefficients) + 25 Emotion Metrics(15 frame-level AUs’ occurrence, 8 frame-level facial expression probabilities as well as frame-level valence and arousal intensities)

## Dataloader
Same dataloader as provided by the Baseline code. We have updated it to load MARLIN files correspondong to each video

## Data Description for our solution

We have used [MARLIN](https://openaccess.thecvf.com/content/CVPR2023/papers/Cai_MARLIN_Masked_Autoencoder_for_Facial_Video_Representation_LearnINg_CVPR_2023_paper.pdf) features of the videos dataset and extracted universal facial representations for our training. These marlin files are present in the Videos directory only with the following nomenclature ```name_of_file_marlin.pt```

This [link](https://drive.google.com/file/d/1DB5mI6U-gS_Q8cD6GesXsSp_Q5h0RtCX/view?usp=sharing) can be used to access a zip file which contains MARLIN files for every videos under the data folder. The structure is same as in the baseline dataset. Copying and pasting in the data directory will write these files in proper manner.

Inside the data folder we have train.csv, val.csv and train_neg.csv. The purpose of train_neg.csv is to consider positive and negative speaker-listener pairs based on the appropriateness matrix given under baseline code.

The example of data structure.
```
data
├── test
├── val
├── train
   ├── Video_files
       ├── NoXI
           ├── 010_2016-03-25_Paris
               ├── Expert_video
               ├── Novice_video
                   ├── 1
                       ├── 1.png
                       ├── ....
                       ├── 751.png
                   ├── 1_marlin.pt
                   ├── ....
           ├── ....
       ├── RECOLA
       ├── UDIVA
   ├── Audio_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.wav
                   ├── ....
           ├── group-2
           ├── group-3
       ├── UDIVA
   ├── Emotion
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.csv
                   ├── ....
           ├── group-2
           ├── group-3
       ├── UDIVA
   ├── 3D_FV_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.npy
                   ├── ....
           ├── group-2
           ├── group-3
       ├── UDIVA
            
```
## Training

For Training from a checkpoint :
python  train.py --batch-size 4 --seq-len 750 --gpu-ids 0  -lr 0.00001  -e 10  -j 4  --outdir results/emotions_marlin_t--max-seq-len 751 --use-video  --contrastive  --resume ./results/emotion_marlin/best_contra_checkpoint.pth

## Evaluation

