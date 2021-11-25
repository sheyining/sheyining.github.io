---
layout: page
title: Intelligent Player
description: Course project of <b>"Artificial Intelligence"</b>. Use <b>Deep Q-learning Network</b> to train an AI Player.
img: assets/img/cs181-ai/ai_cover.png
importance: 3
category: Course Project
---

## Overview

The team project completed in the course **Artificial Intelligence** at the end of the first semester of my sophomore year in *ShanghaiTech* (2019 Fall). Inspired by the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), we intended to implement **Q-Learning** and **Deep Q-Network** to play the pixel game BB-TAN. 

<br/>

## Data Preprocessing

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

<br/>

Since the input of the DQN is screenshots of the game, in order to get rid of redundant elements in the games (e.g. numbers inside squares, pause buttons, illustrations, balls counting text), we made some changes to simplify the original game.

<div class="row">
    <div class="col">
        {% responsive_image path: assets/img/cs181-ai/simple-game.jpg title: "example image" class: "img-fluid" %}
    </div>
</div>
<div class="caption">
    The process of simplifying.
</div>

<br/>

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



## Our Results

Unfortunately, the final result of BBTan AI was not good. It is probably because of the complexity of the BBTan game and also our lack of engineering experience at that time (we were fresh sophomores). 

However, we did make some progress of training a game AI of Break-Out with the DQN.

<div class="row">
    <div class="col">
        <video id="video" controls="" preload="none" >
        <source id="mp4" src="/assets/img/cs181-ai/200.mp4" type="video/mp4">
        </video>
        <div class="caption">
            After 200 episodes.
        </div>
    </div>
    <div class="col">
        <video id="video" controls="" preload="none" >
        <source id="mp4" src="/assets/img/cs181-ai/5000.mp4" type="video/mp4">
        </video>
        <div class="caption">
            After 5000 episodes.
        </div>
    </div>
</div>
