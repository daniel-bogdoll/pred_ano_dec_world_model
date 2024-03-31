## HF2-VAD
Official repository for "Hybrid detection of temporal anomalies for autonomous driving", a comprehensive study demonstrating the application of the HF2-VAD framework to the domain of autonomous driving for enhanced anomaly detection capabilities.

[\[Paper\]
[\[Supp\]]
[\[arXiv\]]

![pipeline]
## 1. Dependencies
```
python==3.6
pytorch==1.5.1
mmcv-full==1.3.1
mmdet==2.11.0
scikit-learn==0.23.2
edflow==0.4.0
PyYAML==5.4.1
tensorboardX==2.4
```
## 2. Usage
### 2.1 Data preparation
Please follow the [instructions](./pre_process/readme.md) to prepare the training and testing dataset.

### 2.2 Train
We train the ML-MemAE-SC at first, then train CVAE model with the reconstructed flows,
and finally finetune the whole framework. All the config files are located at `./cfgs`. 

To train the ML-MemAE-SC, run:
```python
$ python ml_memAE_sc_train.py
```
To train the CVAE model with reconstructed flows, run:
```python
$ python trian.py
```
And finetune the whole HF2VAD framework together as:
```python
$ python finetune.py
```
For different datasets, please modify the configuration files accordingly.

### 2.3 Evaluation
To evaluation the anomaly detection performance of the trained model, run:
```python
$ python eval.py [--model_save_path] [--cfg_file] 
```
E.g., for the ped2 dataset:
```python
$ python eval.py \
         --model_save_path=./pretrained_ckpts/ped2_HF2VAD_99.31.pth \
         --cfg_file=./pretrained_ckpts/ped2_HF2VAD_99.31_cfg.yaml
```
You can download the pretrained weights of HF2VAD for Ped2, Avenue and ShanghaiTech datasets 
from [here](https://drive.google.com/drive/folders/10B7WmZmBSgOPjkbedK9JwH6HRo06VSC2?usp=sharing).

## 3. Results

## Acknowledgment
We thank LiUzHiAn for the PyTorch implementation of the [HF2-VAD](https://github.com/LiUzHiAn/hf2vad).

## Citation
If you find this repo useful, please consider citing:
```
```
