---
title: 'CGAR: A Discrete-Event Simulator for Confidence-Gated Adaptive Routing in RL-Based API Gateways'
tags:
  - Python
  - reinforcement learning
  - multi-armed bandits
  - API gateway
  - microservices
  - cloud computing
authors:
  - name: Khushi Mittal
    affiliation: 1
affiliations:
 - name: Department of Information Technology, Guru Tegh Bahadur Institute of Technology, Guru Gobind Singh Indraprastha University, New Delhi, India
   index: 1
date: 17 July 2026
bibliography: paper.bib
---

# Summary

`CGAR` is a Python discrete-event simulation package for studying how reinforcement-learning (RL) based API gateway routers behave during the exploration period that follows deployment, autoscaling, or a change in the backend pool — the *cold-start problem*. The package implements Confidence-Gated Adaptive Routing (CGAR), a router that maintains a per-backend confidence estimate and uses it to continuously blend a safe stateless heuristic with a learned Q-network policy on a per-backend basis. Because the blend is computed independently for each backend, cold-start protection re-triggers automatically whenever a new backend enters the pool, without resetting the learning already accumulated for the rest of the fleet.

The package includes: a SimPy-based gateway simulator (`simulator.py`) that models client requests, backend latency/queueing/CPU behaviour, and scheduled pool-churn events (adds, removals, replacements); a lightweight NumPy-based Q-network with manual backpropagation (`fast_qnet.py`) built to keep multi-seed experiments runnable on a laptop; seven baseline and ablation routers alongside CGAR itself (`routers.py`), including Round Robin, Least Connections, Pure DQN, epsilon-greedy DQN, and fixed-weight hybrid variants; an experiment runner (`run_experiment.py`) that repeats trials across many random seeds and churn schedules; and plotting utilities (`plot_results.py`) that reproduce the paper's regret, latency, and SLA-violation figures, including a churn-window-versus-steady-state regret decomposition.

# Statement of need

A growing body of work applies deep RL — DQN, PPO, A3C — to API gateway traffic routing and consistently reports gains over static heuristics once the policy has converged [@ghobadi2026adaptive; @xgate2025; @ratelimiting2025]. None of this work evaluates the transient period *before* convergence: immediately after deployment, a scaling event, or a backend pool change, an RL policy must explore by routing live traffic to backends it has not yet evaluated, while a stateless heuristic such as Round Robin or Least Connections never pays that cost. The cold-start problem is well studied in the multi-armed and contextual bandit literature [@auer2002finite; @oetomo2023warm; @bibaut2022robust], but those treatments assume a static, fixed arm set and a single cold-start event — they do not address a routing system whose backend pool changes continuously in production.

`CGAR` gives researchers and practitioners a reusable simulation environment for measuring this gap directly, rather than only reporting post-convergence performance as prior gateway-RL studies do [@ghobadi2026adaptive; @xgate2025; @ratelimiting2025; @li2023reinforcement]. It formalises gateway routing as a non-stationary, per-arm warm-start bandit problem, and ships the churn-window-versus-steady-state regret decomposition needed to isolate cold-start cost from steady-state routing cost. This lets a user reproduce the full experimental pipeline behind CGAR — 20-seed evaluation, four scheduled churn events, seven baselines, and both a floor-bounded adaptive router and its fixed-weight ablations — with a single script, and to substitute their own router, backend-latency model, or churn schedule for further study. The intended audience is researchers working on RL for systems/networking, and engineers evaluating whether a confidence-gated router is worth prototyping as a service-mesh sidecar or Envoy filter before committing to a live rollout.

# Acknowledgements

The author thanks reviewers and collaborators who provided feedback during development.

# References
