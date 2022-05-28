# 🌼OCKD-FacePAD

This repo provides PyTorch implementation of the paper 
[One-Class Knowledge Distillation for Face Presentation Attack Detection](https://arxiv.org/pdf/2205.03792.pdf) to appear on IEEE Transactions on Information Forensics & Security (TIFS).


#  🍀 Data Preparation 

1. Please request and download the datasets. You may use the following address:
 
   🌏 [NTU ROSE-YOUTU](https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/)
   
   🌏 [CASIA FASD](http://www.cbsr.ia.ac.cn/english/FASDB_Agreement/Agreement.pdf)
   
   🌍 [IDIAP REPLAY-ATTACK](https://www.idiap.ch/en/dataset/replayattack)
   
   🌎 [MSU MFSD](https://drive.google.com/drive/folders/1nJCPdJ7R67xOiklF1omkfz4yHeJwhQsz)
   
   🌍 [OULU-NPU](https://www.sites.google.com/site/oulunpudatabase/welcome)


2. Please install [dlib(19.24.0)](https://anaconda.org/conda-forge/dlib) and [opencv(4.5.5)](https://anaconda.org/conda-forge/opencv) in your anaconda environment.
   
3. Please download the pretrained shape predictor model from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2).

4. Please use [preprocessing.py](https://github.com/lizhi-ntu/OCKD-FacePAD/blob/main/preprocessing.py) to get face images from videos. The prepocessed data for client-specific one-class domain adaptation setting are available [here](https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/).

5. Please find the data division of the challenging experimental setting [here](https://github.com/lizhi-ntu/OCKD-FacePAD/blob/main/challenging_setting_data_division.txt).

# 🍀 Teacher Network Training

👀 Please use the example script [train_teacher.py](https://github.com/lizhi-ntu/OCKD-FacePAD/blob/main/train_teacher.py) to train the Teacher Network.

# 🍀 Student Network Training 

👀 Please use the example script [train_student.py](https://github.com/lizhi-ntu/OCKD-FacePAD/blob/main/train_student.py) to train the Student Network.

# 🍀 Model Evaluation

👀 You may use the example script [evaluation_student.py](https://github.com/lizhi-ntu/OCKD-FacePAD/blob/main/evaluation_student.py) to evalute the pretrained model.

# 🍀 Others

👀 Everything in this repo can NOT be used for commercial purpose. 

👀 If you have any questions, feel free to open an issue or contact me via [email](https://github.com/lizhi-ntu/lizhi-ntu/blob/main/README.md).

👀 The implementation of sparse learning in our codes is based on [library](https://github.com/TimDettmers/sparse_learning).

👀 If you use this repo in your work, please use the following citation.

@ARTICLE{9782427, 
author={Li, Zhi and Cai, Rizhao and Li, Haoliang and Lam, Kwok-Yan and Hu, Yongjian and Kot, Alex C.},
journal={IEEE Transactions on Information Forensics and Security}, 
title={One-Class Knowledge Distillation for Face Presentation Attack Detection}, 
year={2022}, 
volume={}, 
number={}, 
pages={1-1}, 
doi={10.1109/TIFS.2022.3178240}
}
