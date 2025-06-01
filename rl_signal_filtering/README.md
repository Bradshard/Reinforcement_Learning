# RL Signal Filtering

A reinforcement-learning–based residual denoiser that augments classical filters (LMS, RLS, Wiener, Sinusoidal Kalman) with a PPO-LSTM agent to learn small corrections on top of the best baseline filter. Includes:

- **`train.py`**: CLI-driven training of a PPO-LSTM residual agent.
- **`evaluate.py`**: CLI or YAML-driven evaluation across multiple noise types/levels, comparing tuned classical filters and the learned model both in SNR and inference time.
- **`src/utils/`**:
  - `signal_utils.py`: signal generation, classical filter implementations, SNR computation.
  - `filter_tuner.py`: grid-search wrapper for LMS, RLS, Wiener, Kalman.
  - `stress_tester.py`: runs full sweeps of signals + filters + PPO.
- **`src/agents/ppo_lstm_agent.py`**: PPO actor-critic model with LSTM.
- **`src/envs/filter_coeff_env.py`**: Gym environment for residual filter coefficient learning.

## Installation

1. **Clone** and **enter** the repo:
   ```bash
    git clone --depth 1 --filter=blob:none --sparse https://github.com/Bradshard/Reinforcement_Learning.git
    cd Reinforcement_Learning
    git sparse-checkout set rl_signal_filtering
    cd rl_signal_filtering
   ```

2. **Create & activate** a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install editable package**:
   ```bash
   pip install -e .
   ```

## Usage

### Training

With a config file:
```bash
python -m rl_signal_filtering.train --config configs/train.yaml
```

Or override parameters:
```bash
python -m rl_signal_filtering.train   --signal_length 2048   --filter_order 5   --num_episodes 500   --lr 1e-4   --alpha 0.3   --beta 0.05
```

Outputs a model `ppo_model.pt`.

### Evaluation

With config:
```bash
python -m rl_signal_filtering.evaluate --config configs/evaluate.yaml
```

Or override:
```bash
python -m rl_signal_filtering.evaluate   --model_path ppo_model.pt   --noise_types gaussian uniform laplacian impulse pink brown   --sigmas 0.1 0.5 1.0 2.0
```

Prints a table of SNR and inference-time comparisons, and plots:
- SNR vs σ line charts
- Inference-time bar chart

## Project Layout

```
rl_signal_filtering/
├── configs/
│   ├── train.yaml
│   └── evaluate.yaml
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── ppo_lstm_agent.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── filter_coeff_env.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── signal_utils.py
│   │   ├── filter_tuner.py
│   │   └── stress_tester.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## Dependencies

- `numpy`
- `torch`
- `gymnasium`
- `pyyaml`
- `tqdm`
- `matplotlib`

Install via:
```bash
pip install -r requirements.txt
```
