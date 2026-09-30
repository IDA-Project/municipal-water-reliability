from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rows = []
    for index in range(50):
        population = 100_000 + index * 1_000
        urban = int(population * 0.6)
        rural = population - urban
        reliability = 40 if index < 15 else 80
        rows.append(
            {
                "Unnamed: 0": "",
                "WSA Name": f"WSA {index:02d}",
                "Population": f"{population:,}".replace(",", " "),
                "Population with access  to water": f"{int(population * 0.9):,}".replace(",", " "),
                "Population Reliable Supply": f"{int(population * reliability / 100):,}".replace(",", " "),
                "% Population Reliable Supply": f"{reliability}%",
                "Urban Population": f"{urban:,}".replace(",", " "),
                "Urban Population with access to water": f"{int(urban * 0.95):,}".replace(",", " "),
                "Urban Population Reliable Supply": f"{int(urban * reliability / 100):,}".replace(",", " "),
                "% Urban Population Reliable Supply": f"{reliability}%",
                "Rural Population": f"{rural:,}".replace(",", " "),
                "Rural Population  with access  to water": f"{int(rural * 0.82):,}".replace(",", " "),
                "Rural Population Reliable Supply": f"{int(rural * reliability / 100):,}".replace(",", " "),
                "% Rural Population Reliable Supply": f"{reliability}%",
            }
        )
    rows[0]["Population with access  to water"] = ""
    return pd.DataFrame(rows)

