# Define your stages
results = {
    "Baseline":    {"L90": 14863.8417, "L98": 15704.9056, "CR": 0.86},
    "MSDSR":       {"L90": 3506.3247,  "L98": 4353.1057,  "CR": 0.41},
    "DDR":         {"L90": 2199.513,   "L98": 2868.774,   "CR": 0.12},
    "DCLS":        {"L90": 1957.366,   "L98": 2619.125,   "CR": 0.01}
}

# Calculate drop rates between stages
stages = list(results.keys())
for i in range(1, len(stages)):
    prev, curr = stages[i-1], stages[i]
    print(f"\nFrom {prev} → {curr}:")
    for metric in ["L90", "L98", "CR"]:
        prev_val = results[prev][metric]
        curr_val = results[curr][metric]
        drop = (prev_val - curr_val) / prev_val * 100
        print(f"  {metric}: {drop:.1f}% reduction")
