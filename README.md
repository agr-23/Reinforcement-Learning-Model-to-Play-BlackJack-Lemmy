# Blackjack Reinforcement Learning Project

**Universidad EAFIT — Artificial Intelligence Course (2025)**  
**Authors:** Jean Carlo Londoño, Alejandro Garcés Ramírez, Maria Acevedo
**Instructor:** Yomin Estiven Jaramillo Múnera  

---

##  Project Purpose

This project explores **reinforcement learning (RL)** methods applied to the game of **Blackjack** under U.S. casino rules (H17, peek enabled).  
Two complementary agents were developed:

- **Tabular Q-Learning Agent** — Implements a discrete-state RL baseline that learns directly from experience.  
- **Deep Q-Network (DQN) Agent** — Employs a neural network with heuristic and supervised pretraining using real hand data.

The main objective is to analyze how **supervised pretraining**, **heuristic guidance (Hi-Lo counting)**, and **deep learning architectures** affect learning efficiency and performance compared to classical RL.

---

##  Key Features

-  Custom **Blackjack environment** with full game logic and U.S. rules.  
-  Two reinforcement learning approaches:
  - **Q-Learning** (discrete state space)
  - **DQN** (continuous state space with replay buffer and soft target updates)
-  **Heuristic integration** with Hi-Lo card counting for improved decision-making.  
-  **Pretraining** on a dataset of historical Blackjack hands (up to 5M records).  
-  **Comprehensive evaluation and logging** for reproducible results.

---

##  Algorithms Overview

### 🔹 Q-Learning (Tabular Baseline)

**State representation:**  
`(player_total, usable_ace, dealer_up, can_double, can_split)`

**Actions:**  
`0 = Hit, 1 = Stand, 2 = Double, 3 = Split`

**Learning rule:**
\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
\]

**Exploration:** ε-greedy policy with linear decay.  
**Goal:** Establish a baseline performance around **40–50% win rate**.

---

###  Deep Q-Network (DQN)

- Three fully connected layers: **512–512–256** (ReLU, Xavier initialization)  
- Soft target network updates: **τ = 0.005**  
- **Loss function:** Mean Squared Error between Q-policy and target network  
- **Pretrained** on a labeled dataset of optimal decisions, then fine-tuned via RL  
- **Heuristic-guided exploration** based on Hi-Lo true count values  

---

##  Repository Structure
<img width="720" height="386" alt="image" src="https://github.com/user-attachments/assets/7400d6a4-6873-4f87-907d-059973a167f4" />


## Installation & Dependencies

### Requirements

- Python ≥ 3.8

- PyTorch ≥ 2.0

- NumPy ≥ 1.23

- Matplotlib ≥ 3.7

- Pandas ≥ 2.0

### Install Dependencies

**pip install -r requirements.txt**

or manually:

**pip install torch numpy matplotlib pandas**

 ## Running the Project
 
**Train Tabular Q-Learning**

python agents/qlearning.py train --episodes 200000 --seed 7 --save models/qtable.pkl

**Evaluate Q-Learning Policy**

python agents/qlearning.py eval --episodes 20000 --load models/qtable.pkl

 **Train Deep Q-Network (DQN)**
python agents/train_dqn.py --episodes 300000 --batch-size 256 --lr 1e-4 \
--buffer-size 500000 --target-update 1 --eps-decay-steps 500000 \
--eval-every 1000 --save-every 1000

**Compare Models**

python agents/compare_models.py

## Dataset for Pretraining

A supervised pretraining phase was implemented using a large dataset of historical Blackjack decisions.
The dataset is publicly available here:

https://drive.google.com/file/d/1w6_zPXF0-cvOJmcVdldVr-AQ_NHF-kks/view?usp=sharing

This dataset allows the neural network to learn initial patterns of optimal play before reinforcement fine-tuning, significantly improving convergence speed.

## Results Summary

| **Model**                  | **Win Rate (%)** |  **Avg. Reward** |  **Episodes** |
|:------------------------------|:-------------------:|:------------------:|:----------------:|
| **Q-Learning**                | 38–41               | −0.09              | 200,000          |
| **DQN (no heuristic)**        | 37                  | −0.08              | 300,000          |
| **DQN + Hi-Lo + Pretraining** | **41**              | **+0.03**          | 300,000          |


The hybrid DQN showed the best convergence and stability.

Despite limited training time and hardware constraints, the architecture proved scalable and efficient for extended training (millions of episodes).

## Expected Results

- Reinforcement learning agents capable of achieving at least 37–42% win rate in Blackjack.

- Demonstration of how heuristic integration and supervised pretraining accelerate learning.

- A modular RL framework reusable for other games or decision-based simulations.

## Notes

- The tabular model excludes the True Count feature to maintain table size.

- The DQN model integrates True Count normalization and basic strategy guidance dynamically.

- Training DQN for millions of episodes requires GPU acceleration (CUDA recommended).

 ## References

- Baldwin, R. et al. “The Optimum Strategy in Blackjack.” Journal of the American Statistical Association, 1956.

- Thorp, E. O. Beat the Dealer: A Winning Strategy for the Game of Twenty-One. Random House, 1966.

- Geisler, S. & Hasseler, T. “Reinforcement Learning in Blackjack.” Stanford University, 2005.

- Liu, J. & Spil, B. “Deep Reinforcement Learning in Blackjack with a Full Deck History.” IEEE CoG, 2021.

##  Acknowledgments

The authors thank Universidad EAFIT and Professor Yomin Estiven Jaramillo Múnera for guidance and academic support throughout the project.



