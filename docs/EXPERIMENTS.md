# Experiments and interpretation

[Back to the README](../README.md)

## Algorithms as implemented

### Tabular Q-learning

The trainer initializes a zero-valued float32 table with 500 states and six actions. Epsilon-greedy action selection is followed by:

```text
target = reward + gamma * max_a Q(next_state, a)
Q(state, action) += alpha * (target - Q(state, action))
```

The current code applies this target even on terminal transitions, without explicitly masking the bootstrap term.

### Neural Q-learning (DQN-style)

An integer state indexes a learned embedding. Two ReLU hidden layers map it to six Q-values:

```text
state ID → Embedding(500, 32) → Linear(32, 128) → ReLU
         → Linear(128, 128) → ReLU → Linear(128, 6)
```

Uniform replay batches train the network with Adam, Huber loss, and gradient-norm clipping. The detached target is:

```text
done = terminated or truncated
target = reward + gamma * (1 - done) * max_a Q_online(next_state, a)
```

There is no separate target network or Double DQN update. The same online network supplies both estimates, and time-limit truncations are treated as terminal for this target.

## Default training configuration

These are current source defaults, not a guarantee of settings used for every stored artifact.

| Setting | Q-learning | DQN-style |
| --- | ---: | ---: |
| Episodes | 3,000 | 3,000 |
| Maximum steps per episode | 200 | 200 |
| Training seed argument | 42 | 42 |
| Discount | 0.99 | 0.97 |
| Learning rate | alpha = 0.7 | Adam lr = 0.0005 |
| Epsilon schedule | 1.0 → 0.10 over 1,500 episodes | 1.0 → 0.10 over 1,500 episodes |
| Training rolling window | 50 | 50 |
| Replay capacity | — | 50,000 transitions |
| Batch size | — | 256 |
| Learning starts | — | 2,000 transitions |
| Update interval | Every transition | Every 2 environment steps |
| Gradient clipping norm | — | 10.0 |

Epsilon uses `episode / eps_decay_episodes`, so episode 1 is already slightly below 1.0. From episode 1,500 onward it stays at 0.10.

## Evaluation protocol

Evaluation uses greedy argmax actions without exploration, with a 200-step cap.

| Setting | Q-learning script | DQN script |
| --- | ---: | ---: |
| Episodes | 1,000 | 100 |
| Reset seeds | 50,042–51,041 | 50,042–50,141 |
| Plot rolling window | 200 | 50 |

Default training resets use seeds 43–3,042. Different reset seed ranges do not imply unseen Taxi states: the state space is small and shared. Use the same episode count, reset sequence, environment version, and step limit for a matched comparison.

Tuning also evaluates on seeds starting at 50,042, so its validation episodes overlap with this test sequence. A future held-out evaluation should use a separate sequence after model selection.

## Metrics

| Metric | Definition | Direction |
| --- | --- | --- |
| Return | Sum of episode rewards | Higher |
| Steps | Actions before termination or truncation | Lower, interpreted with success |
| Penalties | Count of rewards equal to −10 | Lower |
| Success | 1 if a terminal transition awards +20, otherwise 0 | Higher |

A rolling mean averages the last `w` episodes, using the available prefix until the window is full. A rolling success of 1.0 means all episodes in the window succeeded.

The project-specific tuning score in [utils.py](../utils.py) is:

```text
score = 20 * mean_success + mean_return - mean_steps - 10 * mean_penalties
```

Return already includes step costs and illegal-action penalties; the score adds further weight to those costs. It is a selection heuristic, not an independent environment reward.

## Stored evidence

### Best-run tuning records

| Agent | Recorded selected configuration | Greedy result |
| --- | --- | --- |
| Q-learning | alpha 0.3; gamma 0.9; final epsilon 0.05; decay 1,500 episodes | Return 8.01; steps 12.99; penalties 0; success 100% |
| DQN-style | lr 0.0005; batch 128; recorded update interval 1; gamma 0.97 | Return 8.01; steps 12.99; penalties 0; success 100% |

Sources: [tabular log](../results/TUNING_Q_LEARNING.TXT), [DQN summary](../results/dqn_tuning/best_run.txt), and [DQN CSV](../results/dqn_tuning/tuning_results.csv). The search scripts use 1,500 training and 100 evaluation episodes. The DQN interval label does not establish its effective setting because of the mismatch below. The stored tabular log's print format also differs from the current script; no generating revision is attached to these artifacts.

### Final training window

The last rows (episode 3,000) of the committed rolling-mean CSVs contain:

| Agent | Return | Steps | Penalties | Success |
| --- | ---: | ---: | ---: | ---: |
| Q-learning | −0.42 | 15.30 | 0.68 | 1.00 |
| DQN-style | 1.28 | 15.58 | 0.46 | 1.00 |

Sources: [Q-learning CSV](../results/train_q_learning/q_learn_train_rolling_means.csv) and [DQN CSV](../results/train_dqn/dqn_train_rolling_means.csv). Under current defaults, each row summarizes a 50-episode window. These are training statistics with exploration, not greedy test scores or whole-run averages.

The records support successful deliveries in the reported runs. They do not establish a robust algorithm ranking, sample-efficiency advantage across seeds, or comparative computational cost.

## Limitations and next steps

1. **Inactive DQN tuning fields.** The search sets `train_every_steps` and `eps_decay_steps`; the trainer reads `train_every_episodes` and `eps_decay_episodes`. The advertised update-frequency sweep is inactive in the current code. Fix the mapping and rerun it before interpreting that axis.
2. **Inconsistent seeding.** DQN seeds Python, NumPy, PyTorch, the action space, and environment resets. Standalone Q-learning only seeds resets. Tabular tuning seeds global generators once before the grid, not separately per trial, and does not seed the action space. A fixed seed argument alone does not reproduce all runs.
3. **Different terminal handling.** Q-learning always bootstraps; DQN masks termination and truncation. Align target semantics and document time-limit handling before extending the comparison.
4. **No target network.** Add a separate target-network variant as a new experiment when comparing against conventional DQN.
5. **Different evaluation budgets and overlapping splits.** Default tests use 1,000 versus 100 episodes; tuning and testing share reset sequences. Use matched budgets and a separate final evaluation sequence.
6. **Individual stored runs.** There are no multi-seed aggregates or confidence intervals. Repeat training across seeds before claiming an advantage.
7. **Incomplete provenance.** Checkpoints, plots, and summaries lack full generating configurations, dependency locks, and revision metadata. Stored best settings differ from current defaults. Save metadata and raw metrics with future runs.

This documentation update corrects the stale `deep.DQN` imports; it does not change learning algorithms or regenerate historical results.
