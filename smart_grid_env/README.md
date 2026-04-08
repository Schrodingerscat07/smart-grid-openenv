---
title: Smart Grid Demand Response
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
base_path: /web
---

# ⚡ Smart Grid Demand Response (OpenEnv)

## 🌍 The Problem Statement

Every city grid faces a major crisis daily: **unpredictable demand vs. intermittent renewables**. 

Imagine it's 6:00 PM on a 45°C day in Delhi. 20 million ACs switch on. Solar drops to zero. Grid frequency plummets. **In 15 minutes, there will be a blackout unless someone acts.** 

Massive power operators like **Adani Power**, **Tata Power**, or the **Power System Operation Corporation (POSOCO) in India** face this exact challenge. They have to balance the electricity passing through the grid to keep it exactly at **50Hz**. If people use more power than electricity generators produce, the frequency drops. If the frequency drops too low, transformers explode, and the entire city goes entirely dark.

Currently, automated systems that try to handle this use complex numeric models. They exist to balance numbers, but they fail to understand real-world context: an algorithm doesn't inherently understand that turning off power to a **Hospital** is a catastrophe, while briefly reducing power to a **Steel Plant** is acceptable. 

---

## 💎 Our Unique Novelty: An Agentic Approach

Demand response isn't a new problem. Existing Reinforcement Learning (RL) environments for power grids (like CityLearn or Grid2Op) use **flat numeric vectors**. They feed agents arrays like `[50.2, 280.3, 45.1]`. 

**This makes them impossible for Large Language Models (LLMs) to understand.** LLMs reason through language, not arbitrary floats.

**Our environment is the very first demand response simulator natively designed specifically for LLM agents.** Here is why our solution is a breakthrough:

### 1. Situation Reports, Not Vectors
Instead of obscure numbers, the agent observation space contains a naturally written **Strategic Briefing**:
> *"⚠️ WARNING: Grid frequency at 49.6Hz and falling. Evening peak in 45 minutes. Solar output declining. Steel plant running at full capacity (80MW reducible by 32MW). Hospital on backup generator — DO NOT curtail."*

### 2. Cascading Failure Mechanics
If grid frequency falls below **49.0Hz**, loads automatically start disconnecting in a brutal cascade. The agent has to think ahead: *"If I don't curtail the factories now, the hospital goes dark in 3 steps."*

### 3. Constrained Ethical Decision-Making
This environment demands that the agent reasons about ethical trade-offs. The grader strictly evaluates the agent on **fairness** and **Critical Infrastructure Protection**. An agent that keeps the grid alive but repeatedly shuts down hospitals or residential areas will spectacularly fail.

---

## 🎮 Explaining Our Solution

The AI operator has two main tools to combat a grid crisis:

1. **Load Curtailment:** Selectively instructing specific facilities (like factories) to lower their power usage.
2. **Battery Energy Storage (BESS):** Managing a massive 50MWh grid-scale battery to charge up during cheap off-peak hours and discharge during emergencies.

The AI must coordinate these actions over an extended timeframe while navigating the weather, consumer demands, and economic cost.

---

## 🏆 Environment Tasks

We have structured 5 distinct scenarios for agents to prove their capability:

*   **Peak Survival (Easy)**
    *   **Length:** 12 Steps (3 Hours).
    *   **Goal:** Survive an evening rush when solar drops and everyone gets home.
*   **Daily Balance (Medium)**
    *   **Length:** 24 Steps (24 Hours).
    *   **Goal:** Maintain stability across day and night while minimizing consumer discomfort.
*   **Extreme Event (Hard)**
    *   **Length:** 48 Steps (48 Hours).
    *   **Goal:** Manage a brutal Heatwave. The agent must carefully balance stability, cost, and fairness.
*   **Monsoon Crisis (Medium-Hard)**
    *   **Length:** 24 Steps (24 Hours).
    *   **Goal:** Manage erratic wind speeds and zero solar power. Heavy battery management required.
*   **Renewable Transition (Expert)**
    *   **Length:** 72 Steps (3 Days).
    *   **Goal:** Coal plants are retired. Rely strictly on green energy, smart battery cycling, and strict load fairness.

---

## 🕵️ Judging & Testing Guide

If you are evaluating this project, we encourage you to try to "break" the grid using our UI interface to see our physics engine at work:

### The Blackout Challenge
1. In the Web UI, select the **"extreme_event"** task.
2. Click **Reset**.
3. **Do absolutely nothing.** Leave curtailment at `{}` and battery on `idle`.
4. Click **Step** continuously.
5. Notice how Frequency rapidly drops. Once it breaks the threshold, watch the **Situation Report** alert you of cascading failures and blackouts as you receive a massive penalty.

---

## 📊 Proof of Variance (Data)

A good environment must have variance. We mathematically proved the solvability of this environment. **Incompetent agents crash and burn; intelligent agents succeed.**

*   **Do Nothing Strategy (Incompetent):** Score `0.001` (Instant Blackout)
*   **Basic Heuristics Bot:** Score `0.209` (Poor Stability, High Pain)
*   **Our "Smart Oracle" Baseline:** Score `~0.647+` (Smooth Operations)

**OpenEnv Spec Compliant & Ready.** ⚡
