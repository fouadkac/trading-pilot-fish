import json
from datetime import datetime

# Load the JSON data from file
with open("ohlc_hma_with_zones_validations_3D3.json", "r") as file:
    raw_data = json.load(file)

cleaned_data = []

# Define proper keys (with 'hma' instead of 'volume')
keys = ["open", "close", "low", "high", "timestamp", "hma", "flag1", "flag2"]

for sublist in raw_data:
    for entry in sublist:
        if len(entry) == 8:
            cleaned_entry = dict(zip(keys, entry))
            # Convert Unix timestamp to ISO format
            cleaned_entry["timestamp"] = datetime.utcfromtimestamp(cleaned_entry["timestamp"]).isoformat()
            cleaned_data.append(cleaned_entry)

# Save cleaned data
with open("cleaned_data.json", "w") as outfile:
    json.dump(cleaned_data, outfile, indent=4)
