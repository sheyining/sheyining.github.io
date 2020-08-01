---
layout: post
title:  Virtual to Real Reinforcement Learning for Autonomous Driving
date:   2019-11-03
description: Image translation networks to generate real-world images
---

**Paper:** Download <a href="https://arxiv.org/abs/1704.03952" target="_blank">here</a>  
**Video results:** Watch <a href="https://www.youtube.com/watch?v=Bce2ZSlMuqY" target="_blank">here</a>  
**Authors:** Xinlei Pan, Yurong You, Ziyang Wang, Cewu Lu  
**Published in:** BMVC 2017

### Takeaway message
The paper introduces a realistic translation network that uses 2 conditional GANs to generate realistic images from virtual images. The intermediate images is the segmentation map which is the common ground between the real-world and virtual images. An RL agent was trained on the synthesized realistic images using A3C.

<p align="center">
  <img src="/assets/img/virtual_to_real_RL/image-translation-nw.png" width="720">
</p>

### Motivations
RL algorithms for autonomous driving are trained on simulators. The driving policy trained on simulators doesn't translate well to the real-world due to the discrepancy between virtual images and their corresponding real-world images. But both the types of images have a common parsing structure (segmentation map) which can be exploited to generate realistic images in simulation.

### Proposed Solution
The approach consists of 2 image-to-image translation networks. The first network translates the virtual images to their segmentation maps and the second network translates segmented images into their realistic counterparts. Both the networks are GANs and have the same architecture. An A3C with 9 discrete actions is trained on the synthesized realistic images. The authors claim that this is a seminal work of adapting driving policy trained using simulations to the real-world. 

### Evaluation of the solution
Only the steering angle has been used for measuring error. TORCS was used for experiments. Cityscapes dataset was used to obtain the real-world images.
Comparison has been made with respect to 3 A3C agents trained using 3 different approaches.
- *Oracle:* Training and testing on the same environment.
- *Proposed Method:* Training on syntehsized realistic images from E-track. Testing on Cg-track2. The image translation networks were trained on images from both the training and testing tracks. 
- *Domain randomization:* Training on 10 different virtual environments. Testing on Cg-track2.

### Analysis of the problem, idea, and evaluation
**Pros**
- It could work for simple scenarios such as highway driving.
- It Works better than domain randomization method.

**Cons**
- The lane lines aren't preserved in this approach ( *NOTE: this may be a limitation of pix2pix* ).
- The synthesized real images are not always trust-worthy i.e. appearance road intersections even when they are not present in the ground truth data.
- Difficult for the approach to work for urban driving scenarios.
- The evaluation strategy predicts *only* steering angles and is hand-crafted since the actions are discrete.

### Contributions
In general, the approach can generate a decent representation of road, trees, buildings, sky, grass, and sidewalks. The network architecture of the GANs and the A3C DRL, hyperparameters, and training data details have been provided by the authors.

### Future directions
- Segmentation maps are not unique. Adding info regarding the color and texture of objects. 
- Addition of more complex scenarios such as cars, traffic lights, road-signs, and pedestrians.

### Questions
- What are the advantages of this approach over training an RL agent on the segmentation map? Personally I think that this would be easier to train.