# Stochastic Trajectory Prediction via Conditional Diffusion Models

## Get Started
### 1. Create virtual environment
```
conda create -n ccbda_project python=3.8
conda activate ccbda_project
pip install -r requirements.txt
```
### 2. Prepare data
[Download from google drive](https://drive.google.com/drive/folders/18xDXy6Wok4cdkFjORTNaHNJY14aTJvb6?usp=share_link)
```txt
.
└── argo/
    └── raw/
```
### 3. Generate data path in advance to speed up initialization
Replace with your data root in __config.yaml__ 
``` yaml
data:
    root: ~/Downloads/argo/
```
```bash
python generate_path.py
```

## Docker

`cd` into the project directory and run: 

```
docker run --rm -it \
--gpus all \
-p 8848:8888 \
-p 2232:22 \
-v $(pwd):/home \
-e GRANT_SUDO=yes \
-e JUPYTER_ENABLE_LAB=yes \
--user root \
--name gpu-jupyter \
--shm-size 60G \
softmac/jupyterlab:ccbda
```
Change `--shm_size` value to a value lower than your RAM size.

Add `-v PATH_TO_DATA:/data` if needed.

### Enter container

To Enter container: `ssh root@localhost -p 2232`
password: `root`

---
## Idea

Trajectory prediction is a trending topic recent years. There are various deep learning models designed to do predict the trajectories of vehicles and pedestrians in driving scenes. Most of the competitions require models to predict multiple predictions and focus on evaluating the nearest prediction to the ground truth. 
A typical way to perform trajectory prediction is using Transformer model to fuse the information of vehicles' history trajectories, lane lines, and other information. However, we think of this task as a generative task, and should be solved in a different manner. Those Transformer based model can perform well on statistics overall, but cannot really hold a good variety of generated predictions. If think of the task of generative task, Transformer based models are kind of **auto-regressive** approach, which has been replaced with VAE, GAN, or Diffusion models in image generation recent years.
We propose a Diffusion based trajectory prediction model that can generate predictions with good variety. 

## Application
Trajectory prediction is important for self-driving vehicles because it allows the vehicle to anticipate the movements of other objects on the road, such as other vehicles, pedestrians, and obstacles. This allows the vehicle to make decisions about its own motion, such as planning a safe and efficient path through traffic or avoiding collisions. Accurate trajectory prediction is essential for the safe operation of self-driving vehicles, as it enables the vehicle to respond quickly and appropriately to changing traffic conditions.

### Model
Training target: Whole path
Input/Conditioning encoder: Attention model
Latent: 128 dim
Conditioning: History path, Other cars, Semantic information, Map information
Output decoder: MLP

![](https://i.imgur.com/9suOiaB.png)

## Uniqueness or the comparisons with state-of-the art
#### In previous work
_90% of researches are based on auto-regressive approach._
![](https://i.imgur.com/hZ3iCMq.jpg)

#### In this work
_We employ diffusion model as a method for predicting trajectories, and incorporate semantic traffic information in addition to motion data._

## Dataset
### [Argoverse 2](https://www.argoverse.org/av2.html)
Argoverse 2 Motion Forecasting Dataset: 
>Contains 250,000 scenarios with trajectory data for many object types. This dataset improves upon the Argoverse 1 Motion Forecasting Dataset.


## Reference
[latent-diffusion](https://github.com/CompVis/latent-diffusion)

## Members
* 侯俊宇 311511035 jerry.ee07@nycu.edu.tw
* 洪愷尹 311511036 kaiyin.ee11@nycu.edu.tw
* 呂宗翰 311512006 henrylu.ee11@nycu.edu.tw
