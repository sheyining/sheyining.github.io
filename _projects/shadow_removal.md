---
layout: page
title: Portrait Shadow Manipulation
description: Remove foreign shadows and soften facial shadow in a portrait photo based on GridNet.
img: assets/img/shadow_removal/cover.png
importance: 2
category: Mars
---

## Overview

I implemented a Neural Network to removed the foreign shadow on human face in a portrait image based on the paper [Portrait Shadow Manipulation](https://ceciliavision.github.io/project-pages/portrait.html).

In brief, the method proposed in the paper relies on *a pair of neural networks*—one to
remove foreign shadows cast by external objects, and another to soften facial
shadows cast by the features of the subject and to add a synthetic fill light to
improve the lighting ratio. To train the first network, they constructed a dataset of real-world portraits wherein synthetic foreign shadows are rendered onto the face.

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