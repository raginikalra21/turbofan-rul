
import pandas as pd

# Path to your RAW test file
input_path = "data/raw/test_FD001.txt"

# NASA column format
cols = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
] + [f"s{i}" for i in range(1, 22)]

df = pd.read_csv(input_path, sep=" ", header=None)
df = df.dropna(axis=1)
df.columns = cols

engine_id = 18  # CHANGE THIS to export different engines
df_engine = df[df["engine_id"] == engine_id]

output_path = f"engine_{engine_id}.csv"
df_engine.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
