import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def baseline_price(dt):
    """
    Generate a baseline wholesale electricity price in €/kWh.
    Typical real-world wholesale prices:
    - low: 0.03 €/kWh  (30 €/MWh)
    - high: 0.30 €/kWh (300 €/MWh)
    Crisis years (2022–23) slightly higher.
    """

    # --- Year trend: (approximate real-world wholesale progression)
    year_factor = {
        2021: 0.07,   # ~70 €/MWh
        2022: 0.15,   # crisis peak ~150 €/MWh
        2023: 0.11,   # decreased but still elevated
        2024: 0.085   # closer to normalizing
    }[dt.year]

    # --- Seasonal effects
    month = dt.month
    if month in [12, 1, 2]:
        season_mult = 1.25   # expensive winter
    elif month in [6, 7, 8]:
        season_mult = 0.85   # cheaper summer
    else:
        season_mult = 1.05   # shoulder seasons

    # --- Time-of-day load curve (retail-like but plausible for wholesale)
    hour = dt.hour
    if 0 <= hour < 6:
        tod_mult = 0.85
    elif 6 <= hour < 12:
        tod_mult = 1.05
    elif 12 <= hour < 18:
        tod_mult = 1.15
    elif 18 <= hour < 22:
        tod_mult = 1.20   # evening peak
    else:
        tod_mult = 1.00

    return year_factor * season_mult * tod_mult


def generate_prices():
    timestamps = pd.date_range(start="2021-01-01 00:00",
                               end="2024-12-31 23:00",
                               freq="h")

    prices = []
    prev_price = None

    for dt in timestamps:
        base = baseline_price(dt)

        # Small Gaussian noise per hour
        noise = np.random.normal(0.0, 0.002)  # small fluctuations ~0.2 cents

        # Markov smoothing so changes are gradual
        if prev_price is None:
            price = base + noise
        else:
            # 80% previous + 20% new influences
            price = 0.8 * prev_price + 0.2 * (base + noise)

        # Occasional rare spike (crisis moments)
        if np.random.random() < 0.0005:
            price *= np.random.uniform(1.5, 3.5)

        # Ensure positive and clamp to realistic ranges
        price = max(price, 0.02)          # never below 2 cents (20 €/MWh)
        price = min(price, 0.40)          # cap at 40 cents (400 €/MWh)

        prices.append(price)
        prev_price = price


    schema = pa.schema({
        "timestamp": pa.timestamp("ms"),
        "price_per_kwh": pa.float64()
    })
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price_per_kwh": prices
    })

    table = pa.Table.from_pandas(df, schema)
    pq.write_table(table, "test_set_2021-2024.parquet")

    print(f"Generated test_set_2021-2024.parquet with {len(df)} rows.")
    print("Example price range:", df['price_per_kwh'].min(), "to", df['price_per_kwh'].max(), "€/kWh")


if __name__ == "__main__":
    generate_prices()
