# Development and Training of a Reinforcement Learning Model to Play Blackjack (Lemmy)

## Abstract
This repository contains the implementation and experimental framework for the project *"Development and Training of a Reinforcement Learning Model to Play Blackjack (Lemmy)"*.  
The project explores the application of Reinforcement Learning (RL) techniques, including Q-learning and Deep Q-Networks (DQN), to train an autonomous agent capable of mastering the game of Blackjack. The agent integrates classical heuristics such as card counting to improve decision-making under uncertainty.  
This work was conducted within the Artificial Intelligence course at Universidad EAFIT, Medellín, Colombia.

## Objectives
1. Design and implement a Blackjack simulation environment that defines states, actions, and rewards under realistic game rules.  
2. Train and evaluate agents using Q-learning and DQN algorithms.  
3. Integrate heuristic features such as Hi-Lo card counting into the RL training process.  
4. Compare the model’s performance against baseline strategies (basic strategy and random play).  
5. Assess the capability of the trained agent to achieve a win rate above 70% in a simulated environment.

## Methodology Overview
The implementation follows the standard Reinforcement Learning pipeline:
- **Environment definition:** State space includes player total, dealer’s visible card, usable Ace indicator, count value, and decks remaining.  
- **Action space:** Hit, Stand, Double, Split, and Surrender.  
- **Reward structure:** Positive rewards for winning hands, negative rewards for losses, and proportional adjustments for doubling or surrendering.  
- **Training algorithms:** Tabular Q-learning and Deep Q-Network (DQN) with experience replay.  
- **Evaluation:** Comparison with heuristic-based benchmarks and statistical analysis of win rate and average reward.

## State and Action Representation
| Variable | Description | Type / Range |
|-----------|--------------|--------------|
| **Player_Total** | Sum of player’s cards (Ace = 1 or 11) | Integer [2–21] |
| **Dealer_Upcard** | Dealer’s visible card | Integer [1–11] |
| **Usable_Ace** | Indicates if the player has a usable Ace | Binary {0,1} |
| **Count_Value** | Accumulated Hi-Lo counting value | Integer [−20,20] |
| **Decks_Remaining** | Number of decks remaining in the shoe | Decimal [0–8] |
| **Episode_Stage** | Current stage of the episode | Categorical |

| Action | Description | Reward (R) |
|---------|--------------|-------------|
| Hit | Request an additional card | 0 (transition) |
| Stand | Stop and hold the hand | +1 if win, −1 if lose |
| Double | Double the bet and receive one card only | +2 if win, −2 if lose |
| Split | Split a pair into two hands | Depends on each outcome |
| Surrender | Forfeit and lose half the bet | −0.5 |

## Repository Structure
```

Reinforcement-Learning-Model-to-Play-BlackJack-Lemmy/
│
├── agents/
│   └── agent_blackjack.py        # Baseline agent (CLI interaction)
├── src/
│   ├── environment.py            # Blackjack environment
│   ├── train_qlearning.py        # Q-learning agent training
│   └── train_dqn.py              # Deep Q-Network training
├── data/
│   └── dataset.csv               # Generated simulation data
├── logs/
│   └── agent_results.csv         # Training and evaluation logs
├── docs/
│   └── README.md                 # Technical documentation
└── README.md                     # Project summary and report overview

```

## Evaluation Metrics
- **Win rate:** Percentage of games won by the agent.  
- **Average reward:** Expected value of the net reward per hand.  
- **Convergence rate:** Evolution of policy stability across training episodes.

## References
1. R. Baldwin, W. Cantey, H. Maisel, and J. McDermott, “The Optimum Strategy in Blackjack,” *Journal of the American Statistical Association*, vol. 51, no. 275, pp. 429–439, 1956.  
2. E. Thorp, *Beat the Dealer: A Winning Strategy for the Game of Twenty-One*. New York: Random House, 1966.  
3. S. Geisler and T. Hasseler, “Reinforcement Learning in Blackjack,” Stanford University, 2005.  
4. S. S. Dawson, “Reinforcement Learning in the Game of Blackjack,” University of Edinburgh, 2007.  
5. G. Granville, “Applying Reinforcement Learning to Blackjack,” University of Bristol, 2009.  
6. J. Liu and B. Spil, “Deep Reinforcement Learning in Blackjack with a Full Deck History,” *IEEE Conference on Games (CoG)*, 2021.  
7. Y. Buramdoyal and T. R. Gebbie, “The Impact of Deck Size on Q-learning Convergence in Blackjack,” *arXiv preprint*, arXiv:2305.12345, 2023.

## Acknowledgment
The authors acknowledge the guidance and support of **Professor Yomin Estiven Jaramillo Munera** and the **Artificial Intelligence course at Universidad EAFIT**, which provided the resources and context for this research.