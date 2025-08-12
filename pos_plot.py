import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
# --- Load data ---
df = pd.read_csv("data/sample_pos_data.csv")

# We'll build a new DataFrame with normalized, synchronized data
max_cells = 50
cell_data = {}

# --- Process each cell individually ---
for cell_id in range(1, max_cells + 1):
    x_col = f"x{cell_id}"
    y_col = f"y{cell_id}"

    if x_col not in df.columns or y_col not in df.columns:
        continue

    xy = df[[x_col, y_col]].copy()

    # Drop rows where either x or y is NaN
    xy_valid = xy.dropna()

    if xy_valid.empty:
        continue

    # Convert to relative position
    origin = xy_valid.iloc[0]
    xy_relative = xy_valid - origin

    # Reset index so all cells start from t=0
    cell_data[f"x{cell_id}"] = xy_relative[x_col].reset_index(drop=True)
    cell_data[f"y{cell_id}"] = xy_relative[y_col].reset_index(drop=True)

# --- Pad all cell tracks with NaN to align lengths ---
max_len = max(len(s) for s in cell_data.values())
df_out = pd.DataFrame()

for key in sorted(cell_data.keys(), key=lambda k: (int(k[1:]), k[0])):  # sort by cell # and x/y
    s = cell_data[key]
    df_out[key] = s.reindex(range(max_len))

# --- Save to new CSV ---
df_out.to_csv("data/normalized_pos_data.csv", index=False)
print("Saved: data/normalized_pos_data.csv")

# Load the normalized data
df = pd.read_csv("data/normalized_pos_data.csv")

# Detect all cell IDs by looking for xN columns
cell_ids = sorted({int(re.findall(r"\d+", col)[0]) for col in df.columns if col.startswith("x")})

print(f"Detected {len(cell_ids)} cells: {cell_ids}")

# Determine lifespan (number of non-NaN steps) for each cell
lifespans = []

for cell_id in cell_ids:
    x_col = f"x{cell_id}"
    y_col = f"y{cell_id}"
    if x_col in df.columns and y_col in df.columns:
        valid = ~(df[x_col].isna() | df[y_col].isna())
        lifespan = valid.sum()
        lifespans.append(lifespan)

# Shortest lifespan of any cell
min_lifespan = min(lifespans)
print(f"Shortest cell lifespan: {min_lifespan} steps")

# Set up color palette
sns.set(style="darkgrid")
colors = sns.color_palette("husl", n_colors=len(cell_ids))

plt.figure(figsize=(10, 10))

# Plot each cell's truncated trajectory
for i, cell_id in enumerate(cell_ids):
    x_col = f"x{cell_id}"
    y_col = f"y{cell_id}"

    x = df[x_col]
    y = df[y_col]

    # Drop NaNs and reset index
    valid = ~(x.isna() | y.isna())
    x = x[valid].reset_index(drop=True)[:min_lifespan]
    y = y[valid].reset_index(drop=True)[:min_lifespan]

    if len(x) == 0:
        continue

    plt.plot(x, y, label=f"Cell {cell_id}", color=colors[i % len(colors)])

# --- Force fixed square bounds centered on (0, 0) ---
plt.axis("equal")
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.title(f"Cell Tracks, n = {max_cells} (Pseudotime)")
plt.xlabel("Lifetime Δx")
plt.ylabel("Lifetime Δy")
#plt.legend()
plt.tight_layout()
plt.show()