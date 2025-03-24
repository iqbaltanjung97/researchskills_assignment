# researchskills_assignment

# Stochastic Multi-Armed Bandits Assignment

**Course Date**: March 7, 2025  
**Contact**: [thomas.kleinebuening@cs.ox.ac.uk](mailto:thomas.kleinebuening@cs.ox.ac.uk)

---

## 📘 Section 1: Stochastic MABs — Theoretical & Conceptual

We assume a Bernoulli bandit with **K arms**, where each arm \( i \)'s reward follows:

\[
\text{Bernoulli}(\mu_i), \quad \mu_i \in [0, 1]
\]

### Assignment 1: Fixed Exploration Strategy

This strategy includes two phases:
1. **Exploration**: Pull each arm uniformly for \( N \) rounds.
2. **Exploitation**: Pull the arm with the highest sample average.

\[
i_t = \arg\max_{i \in [K]} \hat{\mu}_i^N
\]

#### Questions:
1. Design a MAB instance and show that if \( N = O(1) \), the expected regret is **linear**:  
   \[
   R_T \geq c \cdot T \quad \text{for some constant } c > 0
   \]
2. Discuss how to choose \( N \). What prior knowledge (e.g. \( K \), \( T \), optimality gaps \( \Delta_i \)) helps?
3. Can you choose \( N = f(T, \Delta_i) \) such that the regret is **sublinear**? Prove your bound. *(Hard)*

---

### Assignment 2: ε-Greedy Algorithm

In each round:
- With probability \( \varepsilon \), **explore** (random arm).
- With probability \( 1 - \varepsilon \), **exploit** (arm with highest average).

#### Questions:
1. Design a MAB instance and show that if \( \varepsilon = 0 \), regret is **linear**:
   \[
   R_T \geq c \cdot T
   \]
   *(Assume that each arm is pulled once in the first \( K \) rounds.)*

---

## 🧪 Section 2: Stochastic MABs — Experiments

### Assignment 3: Complete the Code and Answer

#### 1. Exploration vs. Exploitation Tradeoff
- How do the different algorithms balance exploration and exploitation?
- Why is exploration important, and how does too little/much affect regret?

#### 2. Epsilon Schedules for ε-Greedy
- Implement:
  - \( \varepsilon_t = \frac{1}{t} \)
  - \( \varepsilon_t = \frac{1}{\sqrt{t}} \)
  - \( \varepsilon_t = \frac{\log t}{t} \)
- Which schedule gives the best long-term performance? Why?
- How does the decay rate affect convergence?

#### 3. Regret Analysis
- Compare regret curves of all algorithms. Who wins?
- Are results aligned with theory?
- How do hyperparameters (e.g. `c` in UCB, priors in Thompson) affect regret?

#### 4. Thompson Sampling
- How does its regret compare to others?
- How is its exploration different from ε-Greedy and UCB?

#### 5. Sensitivity to Problem Setup
- How do algorithms react to different reward means / gaps \( \Delta_i \)?

#### 6. Robustness to Non-Stationarity
- Suppose the optimal arm changes over time — who adapts best?

#### 7. Computational Complexity
- Are any algorithms impractical with thousands of arms?

---

## 🔁 Section 3: Non-Stationary MABs — Dynamic Pricing

### Assignment 4

Read:  
**"On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems"**  
by Garivier & Moulines (2008)

Complete the code skeleton modelling **dynamic pricing** with non-stationary MABs.

#### Questions:
1. Why do Discounted UCB vs. Sliding Window UCB behave differently?
2. How do hyperparameters (e.g. decay rate \( \gamma \), window size) affect results?
3. What if change points are unknown/irregular?
4. What happens as the number of price levels increases?
5. Implement **gradual changes** in reward instead of abrupt ones. Which algorithm adapts better?

---

## 📚 Recommended Reading

- **Lattimore & Szepesvari** — *Bandit Algorithms* (technical)
- **Slivkins** — *Introduction to Multi-Armed Bandits* (overview)

### Key Papers

- *Finite-Time Analysis of MAB* — Auer et al. (2002)
- *UCB for Exploitation-Exploration Trade-off* — Auer (2002)
- *Non-Stationary MABs* — Besbes et al. (2014)
- *UCB Policies for Non-Stationary Bandits* — Garivier & Moulines (2008)

---

## 🔗 Code Links

- [Stochastic MABs Colab](https://colab.research.google.com/drive/1O3fQf81MQMQ5-Xdwv_NWh4RolQbZENOe?usp=sharing)
- [Non-Stationary MABs Colab](https://colab.research.google.com/drive/1XR9e4PjhFiGPNiikRPxtPkHtASP1V6pF?usp=sharing)