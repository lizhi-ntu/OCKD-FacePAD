# 🌼OCKD-FacePAD

This repo provides PyTorch implementation of the paper 
[One-Class Knowledge Distillation for Face Presentation Attack Detection](https://arxiv.org/pdf/2205.03792.pdf) to be appear on IEEE Transactions on Information Forensics & Security (TIFS).


#  🍀 Data Preparation 

1. Please request and download the datasets. You may use the following address:
 
   🌏 [NTU ROSE-YOUTU](https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/)
   
   🌏 [CASIA FASD](http://www.cbsr.ia.ac.cn/english/FASDB_Agreement/Agreement.pdf)
   
   🌍 [IDIAP REPLAY-ATTACK](https://www.idiap.ch/en/dataset/replayattack)
   
   🌎 [MSU MFSD](https://drive.google.com/drive/folders/1nJCPdJ7R67xOiklF1omkfz4yHeJwhQsz)
   
   🌍 [OULU-NPU](https://www.sites.google.com/site/oulunpudatabase/welcome)


2. Please install [dlib(19.24.0)](https://anaconda.org/conda-forge/dlib) and [opencv(4.5.5)](https://anaconda.org/conda-forge/opencv) in your anaconda environment.
   
3. Please download the pretrained shape predictor model from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2).

4. Please use [preprocessing.py](https://github.com/lizhi-ntu/OCKD-FacePAD/blob/main/preprocessing.py) to get face images from videos. 

# 🍀 Teacher Network Training

👀 Please use the example script [train_teacher.py](https://github.com/lizhi-ntu/OCKD-FacePAD/blob/main/train_teacher.py) to train the Teacher Network.

# 🍀 Student Network Training 

👀 Please use the example script [train_student.py](https://github.com/lizhi-ntu/OCKD-FacePAD/blob/main/train_student.py) to train the Student Network.

# 🍀 Others

👀 The implementation of sparse learning in our codes is based on [library](https://github.com/TimDettmers/sparse_learning).
