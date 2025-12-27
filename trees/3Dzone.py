import json
import time
from typing import List

def load_json_3d(filename: str) -> List[List[List[float]]]:
    with open(filename, 'r') as f:
        return json.load(f)

def save_json_3d(data, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def flatten_3d(data_3d: List[List[List[float]]]) -> List[List[float]]:
    return [bar for sublist in data_3d for bar in sublist]

def is_support(data: List[List[float]], i: int) -> bool:
    return data[i][3] < data[i-1][3] and data[i][3] < data[i+1][3]

def is_resistance(data: List[List[float]], i: int) -> bool:
    return data[i][2] > data[i-1][2] and data[i][2] > data[i+1][2]

def find_pivots(data_flat: List[List[float]], window: int=1):
    supports = []
    resistances = []
    for i in range(window, len(data_flat) - window):
        if is_support(data_flat, i):
            supports.append(data_flat[i][3])
        if is_resistance(data_flat, i):
            resistances.append(data_flat[i][2])
    return supports, resistances

def assign_zones_and_validations(
    data_3d: List[List[List[float]]],
    supports: List[float],
    resistances: List[float]
) -> List[List[List]]:

    flat_data = flatten_3d(data_3d)

    # Construire dictionnaires pour accès rapide par niveau arrondi
    support_dict = {}
    resistance_dict = {}

    def add_to_dict(d, key, value):
        if key not in d:
            d[key] = []
        d[key].append(value)

    for b in flat_data:
        low_rounded = round(b[3], 5)
        high_rounded = round(b[2], 5)
        add_to_dict(support_dict, low_rounded, b)
        add_to_dict(resistance_dict, high_rounded, b)

    support_set = set(round(s, 5) for s in supports)
    resistance_set = set(round(r, 5) for r in resistances)

    output_data_3d = []

    for chunk in data_3d:
        new_chunk = []
        for bar in chunk:
            low = bar[3]
            high = bar[2]
            close = bar[4]

            low_rounded = round(low, 5)
            high_rounded = round(high, 5)

            is_ssl = low_rounded in support_set
            is_bsl = high_rounded in resistance_set

            breakout_bsl = 0
            breakout_ssl = 0

            if is_bsl:
                resistance_levels = resistance_dict.get(high_rounded, [])
                breakout_bsl = int(any(close > r[2] for r in resistance_levels))

            if is_ssl:
                support_levels = support_dict.get(low_rounded, [])
                breakout_ssl = int(any(close < s[3] for s in support_levels))

            # Ajouter les flags zones et validations dans la structure
            new_bar = bar + [[int(is_bsl), int(is_ssl)]] + [[breakout_bsl, breakout_ssl]]
            new_chunk.append(new_bar)

        output_data_3d.append(new_chunk)

    return output_data_3d

def main():
    start = time.time()

    input_file = 'ohlc_hma_time_3D.json'
    output_file = 'ohlc_hma_with_zones_validations_3D.json'

    print("Chargement des données...")
    data_3d = load_json_3d(input_file)
    flat_data = flatten_3d(data_3d)

    print("Détection des zones BSL / SSL...")
    supports, resistances = find_pivots(flat_data)
    print(f"Nombre de supports détectés : {len(supports)}")
    print(f"Nombre de résistances détectées : {len(resistances)}")

    print("Ajout des zones + validations cassures...")
    enriched_data = assign_zones_and_validations(data_3d, supports, resistances)

    print("Sauvegarde dans un nouveau fichier...")
    save_json_3d(enriched_data, output_file)

    print(f"✅ Terminé ! Fichier généré : {output_file} ({time.time() - start:.2f} s)")

if __name__ == "__main__":
    main()
