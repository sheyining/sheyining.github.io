---
layout: page
title: Unity Simulator for Autonomous Vehicle Platoon
description: A first-person driving simulator created in Unity for research of <b>Stable Interaction of Autonomous Vehicle Platoons with Human-Driven Vehicles</b>, advised by Prof. <a href="https://yashpant.github.io/">Yash Vardhan Pant</a> and Prof. <a href="https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/">Zhihao Jiang</a>. Related paper has been submitted to ACC 2022.
img: assets/img/unity_platoon/distracted_photo.png
importance: 3
category: HCPS Lab
---

## Abstract

A necessary prerequisite for the safe interaction of autonomous systems with a human-driven vehicle is for the overall closed-loop system (autonomous systems plus human-driven vehicle) to be stable. This paper studies the safe and stable interaction between a platoon of autonomous vehicles and a set of human-driven vehicles. Considering the longitudinal motion of the vehicles in the platoon, the problem is to ensure a safe emergency braking by the autonomous platoon considering the actions of human-driven vehicles, which may vary based on the driver type. We consider two types of platoon topologies, namely unidirectional and bidirectional. Safe emergency braking is characterized by a specific type of platoon stability, called head-to-tail stability (HTS). We present system-theoretic necessary and sufficient conditions for the combination of the autonomous platoon and human-driven vehicles to be HTS for two platoon control laws, namely the velocity tracking and the platoon formation. Modeling the input-output behavior of each vehicle via a transfer function, the HTS conditions restrict the human-driven vehicles’ transfer functions to have $$H_{\infty}$$  norms below certain thresholds. A safe interaction algorithm first identifies the transfer functions of the human-driven vehicles. Then, it tunes the platoon control gains such that the overall system meets HTS conditions. Theoretical results are validated with both experimental data with human subject studies and simulation studies.

## Unity Driving Simulator

In order to experimentally indentify the human behavior model, I implemented a first-person driving simulator in Unity. Given necessary information, any autonomous platoon and arbitrary driving scenery can be simulated in the virtual environment, and all run-time data can be recorded. 

In the first-person driving game, the driver can use the brake and throttle of a Logitech G29 car controller to control a car (with autonomous steering) that is placed behind an autonomous platoon. The movement of all cars is restricted to a single lane. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/GP46gnF3yY0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
	</div>
<div class="caption">
    A demo in which a human driver modulated the velocity to safely track the vehicle in front. 
</div>


