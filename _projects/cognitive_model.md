---
layout: page
title: Cognitive Digital Twin for Driving Assistance
description: A cognitive digital twin framework that models and learns the driver’s decision process. Advised by Prof. <a href="https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/">Zhihao Jiang</a> and Prof. <a href="https://yashpant.github.io/">Yash Vardhan Pant</a>. Relative paper has been submitted to ICCPS 2022.
img: assets/img/cognitive_model/framework.png
importance: 1
category: HCPS Lab
---

## Abstract

Modern driving assistance systems are equipped with powerful sensors that can better perceive the driving environment compared to human drivers.
The discrepancies in the perception of the driving environment by the driver and the system affect the driver's trust towards the driving assistance system and thus driving safety.
A driving assistance system that infers the driver's decision process can improve the quality of driving assistance as well as the driver's acceptability of the system.
In this paper, we propose a cognitive digital twin framework that models and learns the driver's decision process.
The model captures the driver's perception of the driving environment as well as other agents' driving styles.
Predictions of these perceptions are made based on utility evaluation of future driving environments, which are updated by new observations.
The driver's decision model was implemented as NPC car controller in a virtual driving environment, which was used to validate the decision model and the proposed cognitive digital twin framework.
The mechanisms of the model are first validated by exhibiting expected behaviors in common driving conditions.
The learning algorithm was able to correctly estimate human driver's driving styles and predict the driver's actions using the learned cognitive digital twin.
With the cognitive digital twin, the driving assistance system can better identify risks that the driver is not aware of, and improve the driver's trust towards the driving assistance system.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/cognitive_model/framework.png title: "example image" class: "img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The proposed cognitive digital twin captures thedriver’s decision process. The ADAS can compare the esti-mated driver’s perception to provide appropriate warningsfor the driver.
</div>
