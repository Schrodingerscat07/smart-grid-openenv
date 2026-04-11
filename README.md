---
title: Smart Grid Demand Response
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
base_path: /web
---

> **The first demand response RL environment natively designed for LLM Agents.**  
> Click the **Custom** tab above to open the interactive Control Room.

---

## 🌍 The Problem

India's power grid serves **1.4 billion people** at exactly **50Hz**. If frequency drops even 1Hz below normal, transformers blow and cities go dark.

**Real-world scale of the crisis:**
- **2024 Delhi Heatwave:** Peak demand hit **8,302 MW** — grid operators manually rotated blackouts across 15 districts for 6 hours *(BSES Rajdhani, June 2024)*
- **2022 India Power Crisis:** Coal shortages left **16 of 28 states** with rolling blackouts, affecting **700M+ people** *(IEA World Energy Outlook 2022)*
- **Cost of blackouts:** India loses an estimated **₹1.5 lakh crore ($18B) annually** to unplanned outages *(CEA Report 2023)*

Existing RL environments (CityLearn, Grid2Op) use flat numeric vectors — arrays like `[50.2, 280.3, 45.1]`. **An LLM can't reason about those.** We built the first simulator that speaks natural language.

---

## 💎 How It Works — Simplified

```
┌─────────────┐    Situation Report     ┌──────────────────┐
│  LLM Agent  │ ◄────────────────────── │  Smart Grid Env  │
│  (any model)│                         │                  │
│             │ ──────────────────────► │  Physics Engine  │
└─────────────┘    Action (JSON)        │  10 loads, BESS  │
                                        │  Weather, Freq   │
   "Curtail steel_plant by 15MW         └──────────────────┘
    and discharge battery 20MW"                 │
                                          ┌─────┴─────┐
                                          │ Grader    │
                                          │ Score 0→1 │
                                          └───────────┘
```

**Instead of numbers**, the agent receives a **strategic briefing:**
> *"⚠️ WARNING: Freq at 49.6Hz and falling. Evening peak in 2h. Solar declining. Steel plant at full capacity (80MW, 32MW reducible). Hospital on backup — DO NOT CURTAIL."*

The agent responds with **natural language-style JSON actions**: which loads to curtail, whether to charge/discharge the 100MWh battery. The grader evaluates on **stability × cost × fairness × ethics** simultaneously.

---

## 🏆 The 5 Mission Scenarios

### ⚡ Peak Survival *(Easy — 12 steps)*
**The crisis:** 6 PM Delhi. 20M ACs switch on. Solar drops to zero. 3-hour evening spike.

**How an RL-trained agent saves the day:** The agent pre-charges the battery during afternoon solar surplus, then strategically discharges 50MW during the 6-9 PM peak while curtailing only low-priority factories (steel plant, cement factory) — keeping hospitals and metro running at 100%.

**Real-world impact:** During Delhi's June 2024 heatwave, BSES operators manually rotated 2-hour blackouts. An RL agent could have **eliminated all residential blackouts** by optimally managing the 80MW curtailment window across industrial loads — the same loads that voluntarily participated in India's 2023 demand response pilot.

---

### 📊 Daily Balance *(Medium — 24 steps)*
**The crisis:** Full 24-hour cycle. Balance stability, cost, and consumer comfort across day-night transitions.

**How an RL-trained agent saves the day:** The agent learns the TOU tariff structure (₹6/kWh off-peak → ₹16/kWh super-peak) and shifts industrial curtailments to peak hours where cost savings are 2.7× higher, while charging battery during cheap overnight hours.

**Real-world impact:** India's time-of-use tariff system (introduced by CERC in 2022) saves ₹12,000 crore annually. An RL agent that optimally time-shifts demand response actions could **improve savings by 30-40%** compared to rule-based systems used today.

---

### 🔥 Extreme Heatwave *(Hard — 48 steps)*
**The crisis:** 48-hour heatwave. 45°C+. AC demand surges 35% above normal. Solar output drops in dust haze.

**How an RL-trained agent saves the day:** The agent recognizes cascading failure risk — if frequency drops below 49.0Hz for 2 consecutive steps, loads auto-disconnect. It preemptively curtails low-priority loads 3 steps before the evening peak, preserving frequency above 49.5Hz and preventing a cascade that would have tripped the hospital.

**Real-world impact:** The 2023 North India heatwave caused **150+ heat-related deaths** and grid frequency dropped to 49.16Hz nationally *(POSOCO, April 2023)*. Automated demand response could have maintained frequency above 49.5Hz and **prevented the 47 unauthorized load-shedding events** that affected hospitals.

---

### 🌧️ Monsoon Crisis *(Medium-Hard — 24 steps)*
**The crisis:** Zero solar output. Erratic wind. Heavy reliance on battery and thermal. Waterlogged substations.

**How an RL-trained agent saves the day:** With solar at near-zero all day, the agent aggressively manages the 100MWh battery — charging during low-demand night hours using thermal, then precisely discharging during morning and evening demand spikes. It avoids over-curtailing any single load to maintain the fairness score.

**Real-world impact:** Mumbai's 2020 monsoon grid failure left **20M people without power for 12+ hours** after simultaneous transmission line failures *(BEST Undertaking Report)*. An RL agent managing distributed battery storage could have **reduced outage duration by 70%** through intelligent load prioritization.

---

### 🌱 Renewable Transition *(Expert — 72 steps)*
**The crisis:** Coal plant retired. 100% renewables + battery. 3-day marathon balancing act with weather uncertainty.

**How an RL-trained agent saves the day:** The agent learns to forecast weather transitions (the Markov weather engine) and pre-positions battery state. Before a cloudy-to-storm transition, it charges the battery. Before storm-to-clear, it reduces curtailments. Over 72 steps, it maintains fairness across all 10 loads (Gini coefficient < 0.3).

**Real-world impact:** India's target of **500GW renewable energy by 2030** *(National Electricity Plan 2023)* requires grid operators to manage intermittency without coal backup. This scenario directly trains agents for that future. McKinsey estimates that **AI-optimized grid management could save India $12B annually** by 2030 in reduced curtailment waste and avoided blackouts.

---

## 🛡️ Anti-Exploit Grading

Our graders are hardened against gaming:

| Exploit Attempt | What Happens |
| :--- | :--- |
| Spam same action every step | **20% score penalty** (repetition detection) |
| Curtail hospital/metro repeatedly | **Near-zero score** if >25% of steps |
| Never use the battery | **Miss 5% bonus** (battery diversity reward) |
| Do absolutely nothing | **Cascading failures** → loads auto-disconnect → score collapses |
| Send invalid/garbage inputs | **Silently sanitized** — no crashes, no exploits |

---

## 📋 Environment Variables

| Variable | Description |
| :--- | :--- |
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier for inference |
| `HF_TOKEN` | Hugging Face / API key |

---

**Built for the [Meta PyTorch Hackathon × Scaler](https://pytorch.org/) — OpenEnv Track.** ⚡
