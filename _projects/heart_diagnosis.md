---
layout: page
title: Model-checking-based Diagnosis Assistance for Cardiac Ablation
description: A model-checking-based diagnosis assistance system that doesn't need domain-specific rules. Advised by Prof. <a href="https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/">Zhihao Jiang</a> and Prof. <a href="https://eskang.github.io/">Eunsuk Kang</a>. Relative paper has been submitted to TCPS 2021.
img: assets/img/heart_diagnosis/conduction.png
importance: 2
category: HCPS Lab
---

### Abstract

Cardiac ablation is the primary therapy for tachycardia, which relies on accurate and efficient diagnosis of the patient’s heart condition.  Due to the limitations on the invasiveness of the diagnosis methods, physicians can only infer the patient’s heart condition using information from partial observations from the patient. Due to the complexity and variability of human physiology, there may exist multiple heart conditions that can explain historical observations. The physicians have to enumerate all possible heart conditions that can explain historical observations, and update the set of heart conditions using information from the new observations. The number of suspected heart conditions can be large, which poses heavy mental burden to the physicians, and affects the rigorousness and efficiency of the diagnosis. In this paper, a model-checking-based diagnosis assistance system is proposed to improve accuracy and efficiency of diagnosis in cardiac ablation. Heart models are used to represent the ambiguities during diagnosis. Heart model parameters are refined in order to enumerate suspected heart conditions that may explain the observations. Model checking is used to check whether a heart model can produce historical observations, and proof traces are visualized to the physicians. The system separates computationally-heavy tasks from iterative diagnosis, which can reduce the mental burden of physicians and improve the rigorousness and efficiency of cardiac ablation. The soundness and completeness of the system are rigorously proved, and the efficiency of the system is compared to a previously-proposed rule-based assistance system using clinical case study.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/heart_diagnosis/overview.png title: "example image" class: "img-fluid" %}
    </div>
</div>
<div class="caption">
    1) Electrical activities within the heart; 2) Catheters in the heart during cardiac ablation; 3) Electrogram (EGM) signals sensed from the catheters; 4) Diagnosis Assistance System along with the current setup during cardiac ablation.
</div>

<br/>


### Method

<div class="row">
    <div class="col-sm mt mt-md-4"></div>
    <div class="col-sm mt mt-md-8">
        {% responsive_image path: assets/img/heart_diagnosis/problem.png title: "example image" class: "img-fluid" %}
    </div>
    <div class="col-sm mt mt-md-4"></div>
</div>

EGM observation sequence $$w^o_{GT}$$ can be explained by multiple heart behavior traces (green stars), which correspond to even more parameter vectors (blue squares). The parameter ranges of heart models are refined as more observations occur, so that all feasible behavior traces that can explain EGM sequence can be uniquely identified.

<br/>

<div class="row">
    <div class="col-sm mt-2 mt-md-0"></div>
    <div class="col-sm mt-8 mt-md-0">
        {% responsive_image path: assets/img/heart_diagnosis/parameter.png title: "example image" class: "img-fluid" %}
    </div>
    <div class="col-sm mt-2 mt-md-0"></div>
</div>

Heart model parameters are iteratively refined. Heart models that cannot generate the observed EGM sequences are eliminated. Heart models with the same proof trace are merged to reduce the complexity.

<br/>


### Case Study - Reentry Circuit

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/heart_diagnosis/avnrt.png title: "example image" class: "img-fluid" %}
    </div>
</div>

The mechanism for reentry:
0. depolarization waves conflict in the slow path during normal sinus rhythm; 
1. premature depolarization wave is blocked in the fast path; 
2. depolarization wave enters the fast path from the opposite direction; 
3. depolarization wave traverse around the reentry circuit.

<br/>


### Results

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/heart_diagnosis/result.png title: "example image" class: "img-fluid" %}
    </div>
</div>
<div class="caption">
    Visualization of two proof traces returned from two heart models with two topologies. $$\gamma_2$$ was eliminated after observing the $$5^{th}$$ observation.
</div>

<br/>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/heart_diagnosis/ambiguities.png title: "example image" class: "img-fluid" %}
    </div>
</div>
<div class="caption">
    8 different proof traces of $$\gamma_1$$ returned by the model checker after the $$7^{th}$$ observation. Note that some traces look the same but the event order in different branches are different.
</div>

<br/>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/heart_diagnosis/statistics.png title: "example image" class: "img-fluid" %}
    </div>
</div>
<br/>
Fewer heart models need to be checked if the parameter ranges for heart models increases. Heart models with larger parameter ranges may return traces that are not physiologically-feasible. The rule-based approach checks more heart conditions, but the complexity for each check is low. Rule-based approach is also less likely to have false-positives.