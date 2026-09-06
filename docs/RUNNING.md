# Running the project

[Back to the README](../README.md)

## Installation

Follow the [quick start](../README.md#quick-start). Use Python 3.10+ and run scripts from the repository root: result paths are relative to the current directory. Gymnasium 1.1.1 retains Taxi-v3; the dependency file is not a reconstruction of the original experiment environment. No rendering extras are needed for numeric training or evaluation.

## Commands and outputs

| Command | Input | Output |
| --- | --- | --- |
| `python tabular/test_q_learning.py` | `results/train_q_learning/Q_seed42.npy` | Console summary and four `test_*_1000.png` plots in `results/test_q_learning/` |
| `python dqn/test_dqn.py` | `results/train_dqn/dqn_seed42.pth` | Console summary and four `test_*_100.png` plots in `results/test_dqn/` |
| `python tabular/train_q_learning.py` | Config and main-block constants | Q-table, eight full/zoom plots, and `q_learn_train_rolling_means.csv` in `results/train_q_learning/` |
| `python dqn/train_dqn.py` | Config and main-block constants | Weights, eight full/zoom plots, and `dqn_train_rolling_means.csv` in `results/train_dqn/` |

The `test_*.py` files evaluate RL policies; they are not unit tests. Scripts use fixed output paths and overwrite corresponding files. Preserve artifacts you need before rerunning. DQN reports training progress every 500 episodes.

## Changing a run

Edit [config.py](../config.py) for learning settings, and check the script's main block:

- Training scripts set `SEED`, `EPISODES`, and `OUTDIR` locally; `EPISODES` overrides the dataclass default.
- Evaluation scripts set environment ID, seed, episode count, step limit, checkpoint path, and smoothing window locally.
- DQN evaluation's `EMBEDDING_DIM` and `HIDDEN_DIM` must match the checkpoint architecture.
- Despite its name, `DQNConfig.train_every_episodes` controls the interval in **environment steps** through the trainer's global step counter.

For a matched 1,000-episode evaluation, set `TEST_EPISODES = 1000` in `dqn/test_dqn.py`. Its plot filenames and titles are hardcoded to `100`; update these labels as well when saving the new plots. Q-learning already evaluates 1,000 episodes. Both use reset seeds starting at 50,042.

## Hyperparameter search

These commands run many training trials and are separate from the quick start:

```bash
python tabular/hp_qlearning_tuning.py
python dqn/hp_dqn_tuning.py
```

The tabular search has 108 combinations with 1,500 training and 100 greedy evaluation episodes per combination. It prints the best configuration but does not save the best table or a CSV. To retain the console output:

```bash
python tabular/hp_qlearning_tuning.py > qlearning_tuning.log
```

The DQN search enumerates 36 combinations with the same episode budgets. It overwrites `results/dqn_tuning/tuning_results.csv` and writes `best_run.txt`, but does not save the best weights. Its `train_every_steps` and `eps_decay_steps` assignments are unused by the trainer: read the [limitations](EXPERIMENTS.md#limitations-and-next-steps) before interpreting the update-frequency sweep.

Search results are not automatically applied to `config.py`. Copy the selected settings and retrain if you need a checkpoint for that configuration.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Missing dependency | Activate the virtual environment and run `python -m pip install -r requirements.txt`. Shared utilities require PyTorch even for Q-learning. |
| Taxi-v3 unavailable | Check `python -m pip show gymnasium` and install the pinned dependency file. Keep the environment version consistent when comparing stored results. |
| Checkpoint not found | Run from the repository root and check the input paths above. Train the agent if its checkpoint is absent. |
| DQN checkpoint size mismatch | Match evaluation embedding/hidden dimensions to the training architecture. |
| `No module named 'deep'` in an older checkout | The model lives in `dqn/`. This update corrects training and evaluation imports to `from dqn.DQN import ...`. |
| CUDA unavailable | CPU execution is supported and selected automatically. |

## Artifact provenance

The committed results are historical artifacts. The repository does not include a complete historical dependency lockfile, raw evaluation data, or the script that produced `results/overlays/`. Training CSVs can support new overlay plots, but no overlay-generation command is provided. Record package versions, configuration, seeds, device, and revision with future runs.

## Documentation verification

The updated entry points were checked on Windows with Python 3.12.14, Gymnasium 1.1.1, NumPy 2.3.5, Matplotlib 3.11.1, and PyTorch 2.14.0 (CPU). Both included checkpoint evaluators completed: Q-learning reported return 7.91 and 100% success over 1,000 episodes; DQN reported return 8.01 and 100% success over 100 episodes. These different budgets are not a matched benchmark.

Short three-episode training checks exercised both trainers, including DQN replay updates. Both tuning entry points imported successfully. Full training and grid searches were not rerun; historical artifacts were preserved.
