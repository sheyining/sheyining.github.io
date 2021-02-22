---
layout: post
title: Security Analysis of Safe and Seldonian Reinforcement Learning Algorithms
date: 2021-02-03
venue: NeurIPS-20
topic: SafeRL
description: Analysis of Seldonian RL algorithms in the presence of adversarial attacks 
---

**Paper:** Download <a href="https://papers.nips.cc/paper/2020/hash/65ae450c5536606c266f49f1c08321f2-Abstract.html" target="_blank">here</a>  
**Authors:** A. Pinar Ozisik et. al  
**Published in:** NeurIPS 2020

### Takeaway message
This paper analyzes the robustness of Seldonian RL algorithms against the presence of corrupted trajectories in a safety test dataset, *D*, kept aside for performing a safety test. *D* is created by collecting a set of trajectories by executing a stochastic baseline policy. The safety test checks whether necessary safety constraints are satisfied with high probability, the performance of a policy in this case. The stochasticity of the baseline policy allows estimation of the performance of an evaluation policy using importance sampling (IS) and weighted IS (WIS) w.r.t. *D* which forms the backbone of the analysis. In the presence of corrupted trajectories, capping the IS weights by a clipping value dependent on the upper bound of the number of corrupted trajectories and the level of desired security, *alpha*, results in the importance weighted returns of the evaluation policy being *alpha*-secure. The clipping value is deduced using concentration inequalities, specifically the Chernoff-Hoeffding (CH) inequality in this work, since they lower bound the expectation of the importance weighted returns.

### Motivations
Errors and missing values in *D* can cause the safety test to fail. Specifically, this paper focuses on the worst case scenario where an attacker adds *k* fabricated trajectories to *D* to maximize the estimated performance of the evaluation policy. The optimal attack strategy for a given *k* is to create a trajectory that maximizes the value of the IS weight and return. The largest possible artificial increase in the lower bound of the evaluation policy performance is denoted as *alpha*. If the safety test satisfies the Seldonial RL guarantee even when *D* is corrupted by *alpha*, it is referred to as being *alpha*-secure. This work quantifies the *alpha*-security (for IS) and quasi-*alpha*-security (for WIS) of different off-policy performance estimators.

### Proposed Solution
Given that the upper bound on *k* is known, the proposed approach, Panacea, ensures *alpha*-security. Panacea caps the importance weights for all the samples in *D* by a clipping weight, *c*, which depends on the user specified *alpha* and *k*. The CH lower bound increases with the IS weight and return (proof in appendix). 

### Evaluation of the solution
The *alpha*-security of two safety tests, with and without Panacea, were tested on a 3x3 gridworld domain and on diabetes treatment simulation. The states in the diabetes treatment simulation are the patient body reactions, the actions insulin injection doses, and the reward is the penalty of deviating from the optimal levels of blood glucose. Panacea was shown to be orders of magnitude better at not overestimating the lower bound of the importance weighted returns. Using IS the approach worked well for upto *alpha=0.1* while using WIS the approach was robust for upto *alpha*=0.5 in both the domains even when *k*=150. 

### Analysis of the problem, idea, and evaluation
The problem has high real-world significance since data corruption can happen due to noisy sensors. The idea suggested is simple with strong theoretical guarantees and proofs. The results on the diabetes simulator looks promising. The ides is highly relevant for domains that satisfy the 3 assumptions (inferior evaluation policy, absolute continuity, and the Seldonian safety function) listed in the paper to achieve *alpha*-security.

However, I think that restricting the baseline policy to be stochastic severely limits its applicability. Although, this requirement is a must for obtaining IS estimates, many real-world controllers aren't stochastic.

### Contributions
1. The paper proposes *alpha*-security for quantifying the robustness of the safety test of a Seldonian RL algorithm in the presence of data anomalies.
2. It analyzes the security of existing safety test mechanisms using *alpha*-security and find that even if only one data point is corrupted, the high-confidence safety guarantees provided by several Seldonian RL algorithms can be egregiously violated.
3. It proposes a method that is more robust to anomalies in the data, ensuring safety with high probability when an upper bound on the number of adversarially corrupt data points is known. 
