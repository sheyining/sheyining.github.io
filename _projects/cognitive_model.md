---
layout: page
title: Cognitive Digital Twin for Driving Assistance
description: A cognitive digital twin framework that models and learns the driver’s decision process. Advised by Prof. <a href="https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/">Zhihao Jiang</a> and Prof. <a href="https://yashpant.github.io/">Yash Vardhan Pant</a>. Relative paper has been submitted to ICCPS 2022.
img: assets/img/cognitive_model/framework.png
importance: 1
category: HCPS Lab
---

### Overview

Modern driving assistance systems are equipped with powerful sensors that can better perceive the driving environment compared to human drivers. The discrepancies in the perception of the driving environment by the driver and the system affect the driver's trust towards the driving assistance system and thus driving safety. A driving assistance system that infers the driver's decision process can improve the quality of driving assistance as well as the driver's acceptability of the system.

We propose a cognitive digital twin framework that models and learns the driver's decision process. Three parts are included in this framework: 

1. Driver's Cognitive Model
2. Cognitive Digital Twin Identification
3. Intelligent Driving Assist System



### Framework

<div class="row">
    <div class="col-3"></div>
    <div class="col-6">
        {% responsive_image path: assets/img/cognitive_model/framework.png title: "example image" class: "img-fluid" %}
    </div>
    <div class="col-3"></div>
</div>
<div class="caption">
    The proposed cognitive digital twin captures the driver’s decision process. The ADAS can compare the esti-mated driver’s perception to provide appropriate warningsfor the driver.
</div>

<br/>

### Driver's Cognitive Model

Five main parts for our driver cognitive model are:
1. Observe
2. Update
3. Evaluation
4. Prediction
5. Decision

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/cognitive_model/overview.png title: "example image" class: "img-fluid" %}
    </div>
</div>

<br/>

### Virtual Driving Platform

A virtual driving environment was developed in Unity. Given vehicles' initial status, arbitrary driving scenery can be simulated in the virtual environment, and all necessary run-time information can be recorded. The vehicle can be controlled by a Logitech G29 driving controller, so human players can enjoy a realistic driving experience and make decisions close to what they would do in real life. 

<div class="row">
    <div class="col-sm mt-5 mt-md-0">
        {% responsive_image path: assets/img/cognitive_model/simulatorSetup.png title: "example image" class: "img-fluid" %}
    </div>
    <div class="col-sm mt-5 mt-md-0">
        {% responsive_image path: assets/img/cognitive_model/unity.png title: "example image" class: "img-fluid" %}
    </div>
</div>


<br/>

### Results

<div class="row">
    <div class="col-sm mt-5 mt-md-0">
        {% responsive_image path: assets/img/cognitive_model/human-1.png title: "example image" class: "img-fluid" %}
        <div class="caption">
            Different strategies when interacting with cars with different driving style perceptions
        </div>
    </div>
    <div class="col-sm mt-5 mt-md-0">
        {% responsive_image path: assets/img/cognitive_model/case-2.png title: "example image" class: "img-fluid" %}
        <div class="caption">
            Action prediction accuracy using the Highest-Two criteria
        </div>
    </div>
</div>


