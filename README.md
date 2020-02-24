## About
PyTorch code for our paper:

Zhengyu Zhao, Zhuoran Liu, Martha Larson, [**"Towards Large yet Imperceptible Adversarial Image Perturbations with Perceptual Color Distance"**](https://arxiv.org/abs/1911.02466), CVPR 2020.
<p align="center">
  <img src="https://github.com/ZhengyuZhao/color_adversarial/blob/master/figures/figure1.PNG" width='600'>
</p>
Specifically, we propose to strategically relax tight Lp-norm constraints while still maintaining imperceptibility by using perceptual color distance (CIEDE2000).
The resulting large yet imperceptible perturbations lead to improved robustness and transferability.



## Implementation

### Overview

This code contains the implementations of:
 1. A PyTorch's autograd-compitable differentiable solution of the conversion from RGB to CIELAB space and of CIEDE2000 metric,
 2. Two approaches (PerC-C&W and PerC-AL) to creating imperceptible adversarial perturbations with optimization on perceptual color distance,
 3. Evaluation on success rate, robustness and transferability on 1000 [ImageNet-Compatible images](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/dataset).
 
### Requirements
torch>=1.1.0; torchvision>=0.3.0; tqdm>=4.31.1; pillow>=5.4.1; matplotlib>=3.0.3;  numpy>=1.16.4; 

### Download data

Run [this official script](https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/dataset/download_images.py) to download the dataset.

### Experiments
Code for all the experiments along with descriptions can be found in the Jupyter Notebook file ```main.ipynb```.
Detailed parameter settings for the proposed two approach are described in ```perc_cw.py``` and ```perc_al.py```.

### Examples
<p align="center">
  <img src="https://github.com/ZhengyuZhao/color_adversarial/blob/master/figures/figure3_appendix_1.PNG" width='1000'>
</p>
<p align="center">
  <img src="https://github.com/ZhengyuZhao/color_adversarial/blob/master/figures/figure3_appendix_2.PNG" width='1000'>
</p>
<p align="center">
  <img src="https://github.com/ZhengyuZhao/color_adversarial/blob/master/figures/figure3_appendix_3.PNG" width='1000'>
</p>
<p align="center">
  <img src="https://github.com/ZhengyuZhao/color_adversarial/blob/master/figures/figure3_appendix_4.PNG" width='1000'>
</p>
<p align="center">
  <img src="https://github.com/ZhengyuZhao/color_adversarial/blob/master/figures/figure3_appendix_5.PNG" width='1000'>
</p>
<p align="center">
  <img src="https://github.com/ZhengyuZhao/color_adversarial/blob/master/figures/figure3_appendix_6.PNG" width='1000'>
</p>
