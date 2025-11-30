# Cost-Efficient-Large-Language-Model-Serving-via-Mixed-Integer-Linear-Programming
This is a Python based simulation tool used to analyze the cost-effectiveness of different power allocation strategies (MILP optimal solution vs greedy algorithm vs isomorphic allocation) in different GPU availability scenarios.

## 1 Project background 

This script simulates a real-world scenario of computing power scheduling:
- **task requirements** : a variety of AI tasks with different VRAM and compute requirements.
- **hardware resources** : various types of GPUs (such as RTX 4090, A100, H100, etc.) have different prices and performance parameters.
- **target** : compare the total cost of different scheduling algorithms when handling different workload traces.

## 2 Dependencies

The following Python libraries need to be installed to run this Code:

```bash
pip install -r requirements.txt
```
## 3 Code Structure
The script is organized into the following sections:

### Data Configuration
- Real-world Setup: constants for memory utilization (MEM_UTIL), compute scaling, and scaling factors.

- Specs & Inventory: - AVAIL_CONFIGS: Defines GPU counts for 4 different market scenarios.

    - GPU_SPECS: Price, VRAM, and Compute capability for hardware (e.g., RTX 4090, A100).

    - TYPE_SPECS: Resource requirements for 9 different task types.

    - TRACE_CONFIGS: Probability distributions for workload generation.

### Core Logic
- generate_workload(): Creates synthetic request batches based on trace distributions.

- enumerate_configs(): Generates valid placement configurations (packing tasks onto GPUs) considering memory constraints and speedup factors.

### Solvers
- solve_milp_configs(): Uses PuLP to find the mathematically optimal allocation that minimizes total cost.

- solve_greedy_configs(): A faster, heuristic approach that prioritizes the most cost-effective config for the largest tasks.

- solve_homogeneous_configs(): Forces the scheduler to use only one type of GPU (e.g., "Only A100s") to serve as a baseline comparison.

### Execution & Visualization
- run_experiment():

    - Iterates through all availability scenarios, traces, and scales (1000 to 5000 requests).

    - Runs all solvers.

    - Generates and saves comparison bar charts.

## 4 Usage
Simply run the script using Python:

```Bash
python MILP.py
```
## 5. Output
The script will generate console logs showing the calculated costs for each experiment. Upon completion, it will save 4 PNG images in the current directory, corresponding to the availability scenarios:

    1.scenario_1_scarce.png

    2.scenario_2_balanced.png

    3.scenario_3_high_a100.png

    4.scenario_4_rich.png

Each image contains subplots comparing the algorithms across the different workload traces.