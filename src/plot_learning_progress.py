from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# load data from CSV
log_path = Path("logs/agent_results.csv")
df = pd.read_csv(log_path, names=['episode', 'winrate', 'avg_return', 'epsilon'])
# ignore row of evaluation results if present
df = df[df['episode'].diff() > 0]

# verify expected columns exist
expected_cols = {'episode', 'winrate', 'avg_return', 'epsilon'}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns in CSV: {missing}")

# Smooth curves for better visualization
df['winrate_smooth'] = df['winrate'].rolling(window=20000, min_periods=1).mean()
df['return_smooth'] = df['avg_return'].rolling(window=20000, min_periods=1).mean()

# Create plots
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df['episode'], df['winrate_smooth'], label='Winrate (smoothed)', linewidth=2)
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Winrate", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(df['episode'], df['return_smooth'], color='tab:orange', label='Avg Return (smoothed)', linewidth=2)
ax2.set_ylabel("Recompensa promedio", color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Epsilon (optional)
plt.title("Evolution of Q-Learning Agent Training in Blackjack")
plt.grid(True, alpha=0.3)

# Save plot
plt.tight_layout()
plt.savefig("logs/qlearning_training_progress.png", dpi=300)
plt.show()