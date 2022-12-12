# Stochastic Trajectory Prediction via Conditional Diffusion Models

## Our Repository
[Stochastic Trajectory Prediction via Conditional Diffusion Models]()

## Idea

Trajectory prediction is a trending topic recent years. There are various deep learning models designed to do predict the trajectories of vehicles and pedestrians in driving scenes. Most of the competitions require models to predict multiple predictions and focus on evaluating the nearest prediction to the ground truth. 
A typical way to perform trajectory prediction is using Transformer model to fuse the information of vehicles' history trajectories, lane lines, and other information. However, we think of this task as a generative task, and should be solved in a different manner. Those Transformer based model can perform well on statistics overall, but cannot really hold a good variety of generated predictions. If think of the task of generative task, Transformer based models are kind of **auto-regressive** approach, which has been replaced with VAE, GAN, or Diffusion models in image generation recent years.
We propose a Diffusion based trajectory prediction model that can generate predictions with good variety. 

## Application




## Uniqueness or the comparisons with state-of-the art



## Dataset
### [Argoverse 2](https://www.argoverse.org/av2.html)
Argoverse 2 Motion Forecasting Dataset: 
>Contains 250,000 scenarios with trajectory data for many object types. This dataset improves upon the Argoverse 1 Motion Forecasting Dataset.


## Reference
[MID](https://github.com/gutianpei/mid)
[latent-diffusion](https://github.com/CompVis/latent-diffusion)

## Members
* 侯俊宇 311511035 jerry.ee07@nycu.edu.tw
* 洪愷尹 311511036 kaiyin.ee11@nycu.edu.tw
* 呂宗翰 311512006 henrylu.ee11@nycu.edu.tw
