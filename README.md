# CGAR: Confidence-Gated Adaptive Routing

A discrete-event simulator for studying the cold-start problem in reinforcement-learning-based API gateway routers. CGAR maintains a per-backend confidence estimate and blends a safe stateless heuristic with a learned Q-network policy on a per-backend basis, so cold-start protection re-triggers automatically whenever a backend pool changes (autoscaling, redeploys, failures) — without resetting learning for the rest of the fleet.

See [`paper.md`](paper.md) for the full write-up: problem formulation, method, and results across 20 seeds and four scheduled backend-churn events.

## Installation

```bash
git clone https://github.com/khushi-070906/Confidence-Gated-Adaptive-Routing-CGAR-A-Bandit-Theoretic-Approach-to-the-Cold-Start-Problem.git
cd Confidence-Gated-Adaptive-Routing-CGAR-A-Bandit-Theoretic-Approach-to-the-Cold-Start-Problem
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, `simpy`, `numpy`, `matplotlib` [adjust to match your actual dependencies]

## Usage

Run the full experiment (all 8 routers, 20 seeds, 4 churn events):

```bash
python run_experiment.py
```

Results are written to `results/`. Generate the paper's figures from those results:

```bash
python plot_results.py
```

### Running a single router

```python
from simulator import GatewaySimulator
from routers import CGARRouter  # or RoundRobinRouter, PureDQNRouter, etc.

sim = GatewaySimulator(num_backends=5, num_requests=5000, seed=0)
router = CGARRouter(tau=280, lambda_min=0.85)
results = sim.run(router)
```

[Adjust the exact class/function names to match what's really in `simulator.py` / `routers.py`.]

## Repository structure

| File | Purpose |
|---|---|
| `simulator.py` | Discrete-event gateway simulator (SimPy) with scheduled backend-pool churn |
| `environment.py` | Backend/environment dynamics (latency, queueing, CPU) |
| `routers.py` | CGAR and all baseline/ablation routers |
| `fast_qnet.py` | NumPy Q-network with manual backprop (fast multi-seed training) |
| `run_experiment.py` | Runs all routers across seeds and churn schedules |
| `plot_results.py` | Reproduces the paper's regret/latency/SLA figures |
| `results/` | Output directory for experiment results |

## Citation

If you use this software, please cite it — see [`paper.md`](paper.md) / [`CITATION.cff`](CITATION.cff) [add a CITATION.cff if you want GitHub's "Cite this repository" button].

## License

MIT — see [`LICENSE`](LICENSE).

## Contributing

Issues and pull requests are welcome. [Add a couple of sentences on how you'd like contributions handled, or link a CONTRIBUTING.md if you write one.]
