
# Training compact deep learning models for video classification using circulant matrices

This repository contains the code used for the experiments presented in the [paper][paper] 

The repository is a fork of [google's repository][google's repository] and borrows from [Miech et al][miech] 

The experiments was run with TensorFlow 1.8 on POWER8 machines with 4 TESLA v100 GPU. 

The code is released under Apache License Version 2.0. 

[paper]: https://arxiv.org/abs/1810.01140 
[google's repository]: https://github.com/google/youtube-8m 
[miech]: https://github.com/antoine77340/Youtube-8M-WILLOW

## Setup
```
export TRAIN_FOLDER=data/yt8m/frame/ 
export OUT_FOLDER=output_folder 
export N_GPUS=4 
``` 
/!\ The files in the `output_folder` will be erase if you re-run an experiment in the same folder. 

Record the global parameters: 
``` 
export PARAMS="--train_data_pattern=${TRAIN_FOLDER}train*.tfrecord,${TRAIN_FOLDER}validate[A-Za-z]*.tfrecord --frame_features=True --feature_names=rgb,audio --feature_sizes=1024,128 --iterations=15 --sample_random_frames=True --num_gpu=${N_GPUS} --num_readers=100 --batch_size=80 --num_epochs=15 --base_learning_rate=0.0002 --learning_rate_decay=0.8 --data_augmentation=True --n_bagging=10 --moe_num_mixtures=4 --moe_add_batch_norm=True --k_factor=1 --dbof_cluster_size=8192 --fc_hidden_size=512 --fv_cluster_size=64 --netvlad_cluster_size=64 --model=CirulantDiagonalNetwork --start_new_model" 
```

## Experiment 1
![Alt text](imgs/graph_dc_cd.png?raw=true "Title") 

``` 
python3 code/train.py --train_dir=${OUT_FOLDER} --add_dbof=True --video_level_classifier_model=MoeModel ${PARAMS} 
python3 code/train.py --train_dir=${OUT_FOLDER} --add_dbof=True --use_d_matrix=True --fc_dbof_circulant=True --video_level_classifier_model=MoeModel ${PARAMS} 
```

## Experiment 2
![Alt text](imgs/graph_fc_circulant_embeddings.png?raw=true "Title") 

DBoF figure 
``` 
python3 code/train.py --train_dir=${OUT_FOLDER} --add_dbof=True --video_level_classifier_model=MoeModel ${PARAMS}
python3 code/train.py --train_dir=${OUT_FOLDER} --add_dbof=True --fc_dbof_circulant=True --video_level_classifier_model=MoeModel ${PARAMS} 
``` 

NetVLAD figure 
``` 
python3 code/train.py --train_dir=${OUT_FOLDER} --add_netvlad=True --video_level_classifier_model=MoeModel ${PARAMS} 
python3 code/train.py --train_dir=${OUT_FOLDER} --add_netvlad=True --fc_netvlad_circulant=True --video_level_classifier_model=MoeModel ${PARAMS} 
``` 

NetFisher figure 
``` 
python3 code/train.py --train_dir=${OUT_FOLDER} --add_fisher_vector=True --video_level_classifier_model=MoeModel ${PARAMS} 
python3 code/train.py --train_dir=${OUT_FOLDER} --add_fisher_vector=True --fc_netvlad_circulant=True --video_level_classifier_model=MoeModel ${PARAMS} 
```

## Experiment 3
![Alt text](imgs/graph_layers.png?raw=true "Title") 

``` 
python3 code/train.py --train_dir=${OUT_FOLDER} --add_dbof=True --no_audio=True --video_level_classifier_model=MoeModel ${PARAMS}
python3 code/train.py --train_dir=${OUT_FOLDER} --add_dbof=True --no_audio=True --dbof_circulant=True --video_level_classifier_model=MoeModel ${PARAMS} 
python3 code/train.py --train_dir=${OUT_FOLDER} --add_dbof=True --no_audio=True --fc_dbof_circulant=True --video_level_classifier_model=MoeModel ${PARAMS} 
python3 code/train.py --train_dir=${OUT_FOLDER} --add_dbof=True --no_audio=True --video_level_classifier_model=Circulant_MoeModel ${PARAMS} 
```
