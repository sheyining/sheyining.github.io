---
layout: page
title: Model-checking-based Diagnosis Assistance for Cardiac Ablation
description: A model-checking-based diagnosis assistance system that doesn't need domain-specific rules. Submitted to TCPS 2021.
img: assets/img/heart_diagnosis/conduction.png
importance: 2
category: HCPS Lab
---

## Abstract

Cardiac ablation is the primary therapy for tachycardia, which relies on accurate and efficient diagnosis of the patient’s heart condition.  Due to the limitations on the invasiveness of the diagnosis methods, physicians can only infer the patient’s heart condition using information from partial observations from the patient. Due to the complexity and variability of human physiology, there may exist multiple heart conditions that can explain historical observations. The physicians have to enumerate all possible heart conditions that can explain historical observations, and update the set of heart conditions using information from the new observations. The number of suspected heart conditions can be large, which poses heavy mental burden to the physicians, and affects the rigorousness and efficiency of the diagnosis. In this paper, a model-checking-based diagnosis assistance system is proposed to improve accuracy and efficiency of diagnosis in cardiac ablation. Heart models are used to represent the ambiguities during diagnosis. Heart model parameters are refined in order to enumerate suspected heart conditions that may explain the observations. Model checking is used to check whether a heart model can produce historical observations, and proof traces are visualized to the physicians. The system separates computationally-heavy tasks from iterative diagnosis, which can reduce the mental burden of physicians and improve the rigorousness and efficiency of cardiac ablation. The soundness and completeness of the system are rigorously proved, and the efficiency of the system is compared to a previously-proposed rule-based assistance system using clinical case study.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/heart_diagnosis/overview.png title: "example image" class: "img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    1) Electrical activities within the heart; 2) Catheters in the heart during cardiac ablation; 3) Electrogram (EGM) signals sensed from the catheters; 4) Diagnosis Assistance System along with the current setup during cardiac ablation.
</div>

