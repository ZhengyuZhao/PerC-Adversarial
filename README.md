## About
PyTorch code for our arXiv article:

[**Achieving large yet imperceptible adversarial image perturbations with perceptual color distance metric**](https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/dataset/download_images.py)
<p align="center">
  <img src="https://github.com/ZhengyuZhao/color_adversarial/blob/master/figures/figure1.PNG" width='600'>
</p>
We question the commonly agreed assumption that the imperceptibility of adversarial perturbations corresponds to tight Lp-norm constraints in perception-agnostic RGB space.
Specifically, we propose two approaches to strategically relax such tight constraints while still maintaining imperceptibility by using a perceptually uniform color distance metric.
The resulting large yet imperceptible perturbations lead to improved robustness and transferability.
Integrating additional structural information into the proposed efficient approach yields further improvement on imperceptibility of the images that contain the areas with rich textures.


## Implementation

### Overview

This code contains the implementations of:
 1. A PyTorch's autograd-compitable differentiable solution of the conversion from RGB to CIELAB space and of CIEDE2000 metric,
 2. The proposed two approaches (PerC-C&W and PerC-AL) to creating imperceptible adversarial perturbations,
 3. Evaluation results of success rate, robustness and transferability on 1000 images from the [ImageNet-compatible dataset](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/dataset).
 
### Requirements
torch>=1.1.0; torchvision>=0.3.0; tqdm>=4.31.1; pillow>=5.4.1; matplotlib>=3.0.3;  numpy>=1.16.4; 

### Download data

Run [this official script](https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/dataset/download_images.py) to download the dataset.

### Experiments
Code for all the experiments along with descriptions can be found in the Jupyter Notebook file ```main.ipynb```.
Detailed parameter settings and descriptions of the proposed two approach are presented in ```perc_cw.py``` and ```perc_al.py```.
