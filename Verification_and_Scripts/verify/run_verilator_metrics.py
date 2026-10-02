#!/usr/bin/env python3
import subprocess
import re
import sys

def run_verilator_metrics():
    print("================================================================")
    print("[1/2] Running RTL Simulation (make sim)...")
    print("      This will run Verilator to get the exact per-layer cycles.")
    print("      (Please wait, this might take a few minutes...)")
    print("================================================================\n")
    
    # Run simulation
    sim_proc = subprocess.run(["make", "sim"], capture_output=True, text=True)
    if sim_proc.returncode != 0:
        print("Error running 'make sim'. Output:")
        print(sim_proc.stdout)
        print(sim_proc.stderr)
        sys.exit(1)

    sim_output = sim_proc.stdout

    # Parse per-layer cycles
    # Example line: "  [SIM] Dumped outputs/rtl_layer_0.hex (32768 words) at cycle 4953346"
    layer_cycles = {}
    cycle_regex = re.compile(r"Dumped outputs/rtl_layer_(\d+)\.hex.*?at cycle (\d+)")
    
    prev_cycle = 0
    for line in sim_output.splitlines():
        match = cycle_regex.search(line)
        if match:
            layer_idx = int(match.group(1))
            dump_cycle = int(match.group(2))
            layer_cycles[layer_idx] = dump_cycle - prev_cycle
            prev_cycle = dump_cycle

    print("\n================================================================")
    print("[2/2] Running Functional Check (make check)...")
    print("      This will compare RTL output vs PyTorch Golden Reference.")
    print("================================================================\n")
    
    # Use the python binary specified in the Makefile
    python_bin = "/mnt/ssd_eda/python_envs/eda_venv/bin/python3"
    check_proc = subprocess.run([python_bin, "verify/check_gen.py"], capture_output=True, text=True)
    check_output = check_proc.stdout

    # Parse check_gen.py output
    # Example line: "  Layer E1  [PASS ✓] | MAE: 0.00344 (tol: 0.01) | MaxErr: 0.02340 | Corr: 0.9999"
    layer_metrics = {}
    metric_regex = re.compile(r"Layer\s+(\w+)\s+\[(PASS|FAIL).*?\]\s+\|\s+MAE:\s+([\d\.]+).*?MaxErr:\s+([\d\.]+).*?Corr:\s+([\d\.]+)")
    
    layer_map = {
        0: "E1", 1: "E2", 2: "E3", 3: "E4", 4: "E5", 5: "BOT",
        6: "D1", 7: "D2", 8: "D3", 9: "D4", 10: "D5", 11: "OUT"
    }

    for line in check_output.splitlines():
        match = metric_regex.search(line)
        if match:
            layer_name = match.group(1)
            status = match.group(2)
            mae = float(match.group(3))
            max_err = float(match.group(4))
            corr = float(match.group(5))
            layer_metrics[layer_name] = {
                "status": status,
                "MAE": mae,
                "MaxErr": max_err,
                "Corr": corr
            }

    # Print Final Consolidated Report
    print("\n" + "═"*85)
    print(" 📊 VERILATOR METRICS REPORT: PER-LAYER CYCLES, ERROR & FUNCTIONAL CHECK")
    print("═"*85)
    print(f" {'Layer':<6} | {'Status':<8} | {'Cycles Used':<14} | {'MAE (Error)':<12} | {'Max Error':<12} | {'Correlation':<12}")
    print("─" * 85)
    
    total_cycles = 0
    all_pass = True
    for idx in range(12):
        name = layer_map[idx]
        cycles = layer_cycles.get(idx, 0)
        total_cycles += cycles
        
        metrics = layer_metrics.get(name, {"status": "N/A", "MAE": 0.0, "MaxErr": 0.0, "Corr": 0.0})
        status = metrics["status"]
        if status != "PASS":
            all_pass = False
            
        status_str = f"✅ {status}" if status == "PASS" else f"❌ {status}"
        
        print(f" {name:<6} | {status_str:<8} | {cycles:<14,} | {metrics['MAE']:<12.5f} | {metrics['MaxErr']:<12.5f} | {metrics['Corr']:<12.4f}")

    print("─" * 85)
    print(f" ⏱️  TOTAL LATENCY (SIMULATED): {total_cycles:,} Clock Cycles")
    
    if all_pass:
        print(" 🎯 OVERALL FUNCTIONAL CHECK: PASS (Bit-accurate / Highly Congruent)")
    else:
        print(" ⚠️ OVERALL FUNCTIONAL CHECK: FAIL (Mismatch beyond tolerance)")
    print("═"*85 + "\n")

if __name__ == '__main__':
    run_verilator_metrics()
