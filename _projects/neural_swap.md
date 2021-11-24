---
layout: page
title: High-Resolution Neural Face Swapping for Visual Effects
description: Swap the appearance of a target actor and a source actor while maintaining the target actor’s performance using deep neural network.
img: assets/img/neural_swap/facereplace.jpg
importance: 1
category: Mars
---

## Overview

We implemented the neural face-swapping algorithm proposed in the paper [High-Resolution Neural Face Swapping for Visual Effects](https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/), and made our own improvements to it.

In brief, the network is a kind of *Encoder-Decoder Model*. The key idea of the work is the concept of a *shared  latent space*. The encoder encodes an image to the latent space, while the decoders turn a piece of latent code to an image. 

1. During the training process, images from all identities are embedded in a shared latent space using a common encoder, and these embeddings are then mapped back into pixel space using the decoder corresponding to the desired source appearance. 
2. At testing time, we only need to replace the source identity's decoder with the target's, while keep the common encoder unchanged. All other steps are the same with training.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/neural_swap/pipeline.png title: "example image" class: "img-fluid" %}
    </div>
</div>
<div class="caption">
    Single-encoder (green), multi-decoder (red) network architecture. 
</div>
<br/>

The paper's algorithm takes the advantage of progressive training to enable generation of high-resolution images, and by extending the architecture and training data beyond two people, the network can achieve higher fidelity in generated expressions.



## Our Works

We have put much effort into this work, and made many effective changes to the model after numerous experiments and analysis: (1) adding *Batch-Norm Layers* to some proper positions (2) redesign the *Loss function* and etc. 
With our own improvements, (1) the network can have more stable performance when dealing with less-frequent expressions in the training sets (2) the generated results can reserve more details information (e.g. wrinkles on the face, teeth exposed when laughing).

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <div class="youtube">
       <iframe width="100%" class="elementor-video-iframe" src="https://www.youtube.com/embed/CqQME-OIbKY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
</div>
<div class="caption">
    Our swapping result. The left-top video is the target, other videos are swapping results.
</div>



