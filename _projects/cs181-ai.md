---
layout: page
title: Intelligent Player
description: Course project of <b>"Artificial Intelligence"</b>. Use <b>Deep Q-learning Network</b> to train an AI Player.
img: assets/img/cs181-ai/ai_cover.png
importance: 3
category: Course Project
---

## Overview

The team project completed in the course **Artificial Intelligence** at the end of the first semester of my second year in *ShanghaiTech* (2019 Fall). Inspired by the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), we intended to implement **Q-Learning** and **Deep Q-Network** to play the pixel game BB-TAN. 

<br/>

## Preprocessing

<div class="row g-0">
        <div class="col-md-4">
          {% responsive_image path: assets/img/cs181-ai/bbtan-ui.png title: "example image" class: "img-fluid" %}
        </div>
        <div class="caption">
            BB-TAN game GUI. The game is similar to Break-Out, with every level generating new random bricks at the top. Bricks will be destroyed after enough collisions. Game is over when there is a brick hits the bottom, and the goal is to survive as long as possible using an efficient policy.
        </div>
        <div class="col-md-8">
        	BBTan is a ball shooting game whose core idea is to shoot the ball and destroy the numbered square shape bricks. And the game usually set multiple-level of required hitting times for the square brick, for each bricks the number on the square indicates how many times the players need to hit them to destroy them. And there exists some other extra functional circle bricks with some special gain effect. The player need to get as much as possible scores while surviving longer.
        </div>
</div>
BBTan is a ball shooting game whose core idea is to shoot the ball and destroy the numbered square shape bricks. And the game usually set multiple-level of required hitting times for the square brick, for each bricks the number on the square indicates how many times the players need to hit them to destroy them. And there exists some other extra functional circle bricks with some special gain effect. The player need to get as much as possible scores while surviving longer.

## Deep Q-Network Implementation

<div class="row">
    <div class="col">
        {% responsive_image path: assets/img/cs181-ai/pseudocode_ai.png title: "example image" class: "img-fluid" %}
    </div>
</div>
<div class="caption">
    Pseudocode of DQN from the paper.
</div>

<br/>



## Network Architecture

For foreign shadow removal model, a *GridNet* architecture with modifications was employed. *GridNet* can learn both high-level and low-level features with its multi-resolution conv-deconv grid  architecture which can be seen as an extension of the *U-Net*. 

Besides, the model is supervised with a weighted combination of pixel-space L1 loss ($$L_{pix}$$) and a perceptual feature space loss ($$L_{feat}$$). The perceptual loss is computed by processing the images through a pre-trained *VGG-19 network* and computing the L1 difference between extracted features in selected layers.



## Data Acquisition

To synthesize images with foreign shadows, we model images as a linear 
blend between a “lit” image $$I_l$$ and a “shadowed” image $$I_s$$, according to 
some shadow mask $$M$$:

$$I=I_l\circ (1-M)+I_s\circ M$$

<br/>
<div class="row">
	<div class="col-2"></div>
    <div class="col-8">
        {% responsive_image path: assets/img/shadow_removal/shadow_generate.png title: "example image" class: "img-fluid" %}
    </div>
    <div class="col-2"></div>
</div>

<br/>
Since *style-GAN* generated images are lifelike and quite similar to human protrait images, I used it to generate 50,000 different identities and took the method above to synthesize the foreign shadow dataset.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/shadow_removal/train_data.png title: "example image" class: "img-fluid" %}
    </div>
</div>
<br/>

## Results
The testing data was sythesized in the same way with training data, but with different identities and different shadow shapes. 
The final model can perform very well on the testing data, but there will be obvious masks on the real-world shadow portaits. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/shadow_removal/output.png title: "example image" class: "img-fluid" %}
    </div>
</div>
<div class="caption">
    Results on the Sythesized Testing data.
</div>
<div class="row">
    <div class="col">
        {% responsive_image path: assets/img/shadow_removal/output_bad.png title: "example image" class: "img-fluid" %}
    </div>
</div>
<div class="caption">
    Results on the real-world data.
</div>