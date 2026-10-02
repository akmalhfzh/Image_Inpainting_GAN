vcd_file = "generator_top.vcd"

monitored = {"U#": "state", "F#": "fill_c_in"}

with open(vcd_file, "r") as f:
    curr_time = 0
    for line in f:
        line = line.strip()
        if not line: continue
        if line.startswith("#"):
            curr_time = int(line[1:])
        elif line.startswith("b"):
            parts = line.split()
            val = parts[0][1:]
            sym = parts[1]
            if sym in monitored:
                print(f"Time {curr_time:6d} | {monitored[sym]:10s} <= {val}")
