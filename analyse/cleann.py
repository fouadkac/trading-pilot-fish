import pandas as pd

# Try UTF-16 first (common for MetaTrader exports)
df = pd.read_csv("export.csv", sep=";", encoding="utf-16")

# If UTF-16 doesn’t work, try Windows-1252:
# df = pd.read_csv("export.csv", sep=";", encoding="cp1252")

# Then continue cleaning as before
df = df.drop_duplicates(subset=["Time"])
df["Time"] = pd.to_datetime(df["Time"], format="%Y.%m.%d %H:%M:%S")
df = df.sort_values("Time").reset_index(drop=True)
df = df.fillna(method="ffill")

numeric_cols = ["Open","High","Low","Close","HighHighest10","LowLowest10","DiffPrevHighCurrentLow"]
df[numeric_cols] = df[numeric_cols].astype(float)

df.to_csv("export_cleaned.csv", index=False, sep=";")

print("✅ Data cleaned and saved to export_cleaned.csv")
