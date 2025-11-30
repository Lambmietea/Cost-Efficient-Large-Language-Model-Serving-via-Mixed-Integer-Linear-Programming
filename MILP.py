import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

# ==========================================
# 1. Real-world Data Setup 
# ==========================================
MEM_UTIL = 0.90  # memory util fraction
COMP_ADJUST = 10  # compute scaling 
SCALE_FACTOR = 150

AVAIL_CONFIGS = {
    "Avail 1 ": {
        "RTX_4090": 16, "A40": 12, "RTX_A6000": 8, "L40": 12, "A100_80G": 6, "H100": 8
    },
    "Avail 2 ": {
        "RTX_4090": 32, "A40": 8, "RTX_A6000": 16, "L40": 16, "A100_80G": 7, "H100": 12
    },
    "Avail 3 ": {
        "RTX_4090": 32, "A40": 16, "RTX_A6000": 8, "L40": 8, "A100_80G": 32, "H100": 8
    },
    "Avail 4 ": {
        "RTX_4090": 24, "A40": 24, "RTX_A6000": 24, "L40": 16, "A100_80G": 4, "H100": 8
    }
}

GPU_SPECS = {
    "RTX_4090": {"price": 0.53, "memory": 24, "compute": 83},
    "A40":       {"price": 0.55, "memory": 48, "compute": 150},
    "RTX_A6000": {"price": 0.83, "memory": 48, "compute": 91},
    "L40":       {"price": 0.90, "memory": 48, "compute": 181},
    "A100_80G":  {"price": 1.75, "memory": 80, "compute": 312},
    "H100":      {"price": 2.99, "memory": 80, "compute": 989},
}

TYPE_SPECS = {
    1: {"mem_req": 4,  "comp_req": 20},
    2: {"mem_req": 8, "comp_req": 30},
    3: {"mem_req": 14, "comp_req": 40},
    4: {"mem_req": 6,  "comp_req": 35},
    5: {"mem_req": 12, "comp_req": 45},
    6: {"mem_req": 26, "comp_req": 55},
    7: {"mem_req": 10, "comp_req": 90},
    8: {"mem_req": 18, "comp_req": 100},
    9: {"mem_req": 32, "comp_req": 120}
}

TRACE_CONFIGS = {
    "Trace 1 (Swiss AI)": [0.33, 0.07, 0.08, 0.07, 0.27, 0.06, 0.06, 0.03, 0.03],
    "Trace 2 (Azure)":    [0.22, 0.05, 0.05, 0.21, 0.05, 0.05, 0.19, 0.06, 0.12],
    "Trace 3 (WildChat)": [0.04, 0.14, 0.03, 0.20, 0.27, 0.01, 0.25, 0.00, 0.06]
}

def generate_workload(trace_name, num_requests):
    ratios = np.array(TRACE_CONFIGS[trace_name])
    ratios = ratios / np.sum(ratios)
    ids = range(1, 10)
    generated_ids = np.random.choice(ids, size=num_requests, p=ratios)
    agg_workload = {i: 0 for i in ids}
    for gid in generated_ids: agg_workload[gid] += 1
    return agg_workload

def get_gpu_types(avail_name):
    config = AVAIL_CONFIGS[avail_name]
    gpu_types = {}
    for name, qty in config.items():
        gpu_types[name] = GPU_SPECS[name].copy()
        gpu_types[name]["available"] = qty * SCALE_FACTOR
    return gpu_types

# speedup S(k) for k in {1,2,4,8,16}
SCALE_KS = [1, 2, 4, 8]  #typically restrict up to 8 for ops
SPEEDUP = {1: 1.0, 2: 1.75, 4: 3.2, 8: 5.6}  # unified scaling

# ==========================================
# Build configurations: for each GPU type
# ==========================================
def enumerate_configs(gpu_types):
    configs = []
    # compute per-card base throughput
    h_card = {g: {} for g in gpu_types}
    for g, spec in gpu_types.items():
        for w, t in TYPE_SPECS.items():
            # if a single task doesn't fit into memory -> zero
            if t["mem_req"] > spec["memory"] * 0.95:
                h_card[g][w] = 0.0
            else:
                mem_cap = (spec["memory"] * MEM_UTIL) / t["mem_req"]
                comp_cap = (spec["compute"] * COMP_ADJUST) / t["comp_req"] if t["comp_req"] > 0 else float('inf')
                h_card[g][w] = min(mem_cap, comp_cap)
    # enumerate
    cid = 0
    for g in gpu_types:
        for k in SCALE_KS:
            if k not in SPEEDUP: 
                continue
            # cost of one config = k * price_per_card
            cost = gpu_types[g]["price"] * k
            dn = {tg: 0 for tg in gpu_types}
            dn[g] = k
            # throughput per config for each w: h_card[g][w] * SPEEDUP[k]
            h_cw = {}
            for w in TYPE_SPECS:
                h_cw[w] = h_card[g][w] * SPEEDUP[k]
            configs.append({
                "id": f"{g}_x{k}",
                "gpu_type": g,
                "k": k,
                "dn": dn,
                "cost": cost,
                "h_cw": h_cw
            })
            cid += 1
    return configs

# ==========================================
# MILP
# ==========================================
def solve_milp_configs(workload_counts, gpu_types, configs=None, time_limit=120):
    if configs is None:
        configs = enumerate_configs(gpu_types)
    # feasible check: ensure at least one config can host each task type
    for w, cnt in workload_counts.items():
        possible = any(cfg["h_cw"][w] > 1e-12 for cfg in configs)
        if not possible and cnt > 0:
            return float('inf')

    prob = pulp.LpProblem("Config_MILP", pulp.LpMinimize)
    # variables
    y = {cfg["id"]: pulp.LpVariable(f"y_{cfg['id']}", lowBound=0, cat='Integer') for cfg in configs}
    flow = {w: {cfg["id"]: pulp.LpVariable(f"flow_{w}_{cfg['id']}", lowBound=0, cat='Continuous') for cfg in configs} for w in TYPE_SPECS}

    # objective: minimize cost
    prob += pulp.lpSum([y[cfg["id"]] * cfg["cost"] for cfg in configs])

    # assignment
    for w, cnt in workload_counts.items():
        prob += pulp.lpSum([flow[w][cfg["id"]] for cfg in configs]) == float(cnt)

    # capacity per config: flow[w][c] <= y_c * h_cw
    for cfg in configs:
        cid = cfg["id"]
        for w in TYPE_SPECS:
            if cfg["h_cw"][w] <= 1e-12:
                prob += flow[w][cid] == 0
            else:
                prob += flow[w][cid] <= y[cid] * cfg["h_cw"][w]

    # GPU availability: sum_c y_c * k_c * 1_{type==g} <= available[g]
    for g in gpu_types:
        prob += pulp.lpSum([y[cfg["id"]] * cfg["k"] * (1 if cfg["gpu_type"] == g else 0) for cfg in configs]) <= gpu_types[g]["available"]

    # optional small slack aggregated resource constraints could be added, omitted here

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    status = pulp.LpStatus[prob.status]
    if status != 'Optimal' and status != 'Optimal':
        # fallback: if infeasible or solver didn't return objective
        if pulp.value(prob.objective) is None:
            return float('inf')
    val = pulp.value(prob.objective)
    if val is None:
        return float('inf')
    return val

# ==========================================
# Greedy
# ==========================================
def solve_greedy_configs(workload_counts, gpu_types, configs=None):
    if configs is None:
        configs = enumerate_configs(gpu_types)
    # compute per-config throughput sum (total across all task types if used solely for that type)
    # For greedy we pick per-task-type allocations; cost-per-task for config c on task w:
    cost_per_task = {cfg["id"]: {} for cfg in configs}
    for cfg in configs:
        for w in TYPE_SPECS:
            h = cfg["h_cw"][w]
            if h <= 0:
                cost_per_task[cfg["id"]][w] = float('inf')
            else:
                cost_per_task[cfg["id"]][w] = cfg["cost"] / h

    inventory = {g: gpu_types[g]["available"] for g in gpu_types}
    config_inventory = {cfg["id"]: (cfg["k"], cfg["gpu_type"]) for cfg in configs}  # (k, type)
    # but easier: compute how many copies of each config possible given inventory
    max_configs = {}
    for cfg in configs:
        g = cfg["gpu_type"]; k = cfg["k"]
        if gpu_types[g]["available"] // k > 0:
            max_configs[cfg["id"]] = gpu_types[g]["available"] // k
        else:
            max_configs[cfg["id"]] = 0

    total_cost = 0.0
    # process tasks largest mem first
    sorted_tasks = sorted(workload_counts.keys(), key=lambda w: TYPE_SPECS[w]['mem_req'], reverse=True)
    # maintain remaining config inventory (counts)
    remaining_cfg = dict(max_configs)

    for w in sorted_tasks:
        remaining = int(workload_counts[w])
        if remaining <= 0:
            continue
        # candidate configs able to host this task
        candidates = [cfg for cfg in configs if cfg["h_cw"][w] > 0 and remaining_cfg[cfg["id"]] > 0]
        if not candidates:
            return float('inf')
        # sort by cost per task
        candidates.sort(key=lambda cfg: cost_per_task[cfg["id"]][w])
        for cfg in candidates:
            if remaining <= 0:
                break
            cid = cfg["id"]; k = cfg["k"]; gtype = cfg["gpu_type"]
            slots_per_cfg = cfg["h_cw"][w]  # tasks per config copy
            if slots_per_cfg <= 0:
                continue
            # how many copies needed
            need = int(math.ceil(remaining / slots_per_cfg))
            take = min(need, remaining_cfg[cid])
            if take <= 0:
                continue
            remaining_cfg[cid] -= take
            total_cost += take * cfg["cost"]
            remaining -= take * slots_per_cfg
        if remaining > 0:
            return float('inf')
    return total_cost

# ==========================================
# Homogeneous 
# ==========================================
def solve_homogeneous_configs(workload_counts, gpu_types, gpu_name, configs=None):
    if configs is None:
        configs = enumerate_configs(gpu_types)
    configs_g = [c for c in configs if c["gpu_type"] == gpu_name]
    if not configs_g:
        return float('inf')
    best_cost = float('inf')

    for cfg in configs_g:
        cid = cfg["id"]
        prob = pulp.LpProblem(f"Homog_Config_{cid}", pulp.LpMinimize)
        y = pulp.LpVariable("y", lowBound=0, upBound=gpu_types[gpu_name]["available"] // cfg["k"], cat='Integer')
        flow = {w: pulp.LpVariable(f"flow_{w}", lowBound=0, cat='Continuous') for w in TYPE_SPECS}
        # objective: minimize cost = y * cfg.cost
        prob += y * cfg["cost"]
        # assignment
        for w, cnt in workload_counts.items():
            prob += flow[w] == float(cnt)
        # capacity: flow[w] <= y * h_cw
        for w in TYPE_SPECS:
            if cfg["h_cw"][w] <= 1e-12:
                if workload_counts[w] > 0:
                    prob += flow[w] == 0  # infeasible later
                else:
                    prob += flow[w] == 0
            else:
                prob += flow[w] <= y * cfg["h_cw"][w]
        # GPU availability: y * k <= available
        prob += y * cfg["k"] <= gpu_types[gpu_name]["available"]
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
        if pulp.LpStatus[prob.status] == 'Optimal' or pulp.LpStatus[prob.status] == 'Feasible':
            val = pulp.value(y * cfg["cost"])
            if val is not None and val < best_cost:
                best_cost = val
    return best_cost

# ==========================================
# Runner & Visualization 
# ==========================================
def run_experiment():
    print("Experiment: Saving 4 Separate Plots for each Availability Scenario")
    scales = [1000, 3000, 5000]

    file_names = {
        "Avail 1 ": "scenario_1_scarce.png",
        "Avail 2 ": "scenario_2_balanced.png",
        "Avail 3 ": "scenario_3_high_a100.png",
        "Avail 4 ": "scenario_4_rich.png"
    }
    
    blue_palette = ["#B8DBFF", "#8EC0DF", '#6BAED6', '#3182BD', '#08519C']

    for avail_name, file_name in file_names.items():
        print(f"\n>>> Generating Plot for {avail_name}...")
        current_gpus = get_gpu_types(avail_name)
        configs = enumerate_configs(current_gpus)

        # debug: print some config stats
        # for c in configs:
        #     print(c['id'], "k", c['k'], "cost", c['cost'], "h_sample", {w:c['h_cw'][w] for w in c['h_cw']})

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Cost Efficiency Analysis - {avail_name}", fontsize=16, fontweight='bold')

        for cidx, (trace_name, _) in enumerate(TRACE_CONFIGS.items()):
            print(f"   Processing {trace_name}...")
            milp_res, greedy_res, homo_a100_res, homo_a6000_res, homo_4090_res = [], [], [], [], []

            for N in scales:
                wl = generate_workload(trace_name, N)
                milp_cost = solve_milp_configs(wl, current_gpus, configs=configs, time_limit=120)
                greedy_cost = solve_greedy_configs(wl, current_gpus, configs=configs)
                homo_a100_cost = solve_homogeneous_configs(wl, current_gpus, "A100_80G", configs=configs)
                homo_a6000_cost = solve_homogeneous_configs(wl, current_gpus, "RTX_A6000", configs=configs)
                homo_4090_cost = solve_homogeneous_configs(wl, current_gpus, "RTX_4090", configs=configs)

                milp_res.append(milp_cost)
                greedy_res.append(greedy_cost)
                homo_a100_res.append(homo_a100_cost)
                homo_a6000_res.append(homo_a6000_cost)
                homo_4090_res.append(homo_4090_cost)

                print(f"     N={N}: MILP={milp_cost}, Greedy={greedy_cost}, Homo_A100={homo_a100_cost}, Homo_A6000={homo_a6000_cost}, Homo_4090={homo_4090_cost}")

            ax = axes[cidx]
            x = np.arange(len(scales))
            width = 0.15

            valid_vals = [v for v in greedy_res + milp_res + homo_a100_res + homo_a6000_res if v != float('inf')]
            max_val = max(valid_vals) if valid_vals else 1000
            vis_cap = max_val * 1.3
            def cap(v): return min(v, vis_cap) if v != float('inf') else vis_cap

            ax.bar(x - 2*width, [cap(v) for v in homo_a100_res], width, label='Homo(A100)', color=blue_palette[0])
            ax.bar(x - 1*width, [cap(v) for v in homo_a6000_res], width, label='Homo(A6000)', color=blue_palette[1])
            ax.bar(x, [cap(v) for v in homo_4090_res], width, label='Homo(4090)', color=blue_palette[2])
            ax.bar(x + 1*width, [cap(v) for v in greedy_res], width, label='Greedy', color=blue_palette[3])
            ax.bar(x + 2*width, [cap(v) for v in milp_res], width, label='MILP', color=blue_palette[4])

            for i, v in enumerate(homo_4090_res):
                if v == float('inf'): ax.text(x[i], vis_cap*0.5, 'Fail', ha='center', rotation=90, color='black', fontweight='bold', fontsize=8) # Changed text to black for visibility on light bg
            for i, v in enumerate(homo_a6000_res):
                if v == float('inf'): ax.text(x[i] - 1*width, vis_cap*0.5, 'Fail', ha='center', rotation=90, color='black', fontweight='bold', fontsize=8)

            ax.set_title(trace_name, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(scales)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xlabel('Scale (Requests)')
            if cidx == 0:
                ax.set_ylabel('Total Cost ($)')
                ax.legend(loc='upper left', fontsize='small')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(file_name, dpi=300)
        plt.close(fig)
        print(f"Saved {file_name}")

    print("\n All 4  Plots Generated Successfully")

if __name__ == "__main__":
    run_experiment()