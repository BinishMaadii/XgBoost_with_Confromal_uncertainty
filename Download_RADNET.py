"""
Usage:
    python download_radnet.py
"""
import pandas as pd
import time

COLUMNS = {
    "LOCATION_NAME": "LOCATION_NAME",
    "SAMPLE COLLECTION TIME": "SAMPLE_TIME",
    "DOSE EQUIVALENT RATE (nSv/h)": "DOSE_RATE",
    "GAMMA COUNT RATE R02 (CPM)": "R02",
    "GAMMA COUNT RATE R03 (CPM)": "R03",
    "GAMMA COUNT RATE R04 (CPM)": "R04",
    "GAMMA COUNT RATE R05 (CPM)": "R05",
    "GAMMA COUNT RATE R06 (CPM)": "R06",
    "GAMMA COUNT RATE R07 (CPM)": "R07",
    "GAMMA COUNT RATE R08 (CPM)": "R08",
    "GAMMA COUNT RATE R09 (CPM)": "R09",
    "STATUS": "STATUS",
}

# (state, city) as used in the RadNet URL path, and how many years back
# has both gamma-count and exposure-rate data (check the RadNet CSV
# downloads page for each station's start dates before adding one)
STATIONS = [
    ("pa", "philadelphia"),
    ("oh", "cincinnati"),
    ("pa", "pittsburgh"),
    ("ca", "sacramento"),
]

YEARS = range(2018, 2027)  # adjust based on each station's exposure-rate start date

BASE_URL = "https://radnet.epa.gov/cdx-radnet-rest/api/rest/csv/{year}/fixed/{state}/{city}"

def fetch_station_year(state, city, year):
    url = BASE_URL.format(year=year, state=state, city=city)
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"  skip {state}/{city} {year}: {e}")
        return None
    if df.empty:
        return None
    df = df.rename(columns=COLUMNS)
    keep = [c for c in COLUMNS.values() if c in df.columns]
    df = df[keep]
    df = df[df["STATUS"] == "APPROVED"]
    df = df.dropna(subset=["DOSE_RATE", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09"])
    return df


def main():
    frames = []
    for state, city in STATIONS:
        for year in YEARS:
            print(f"Fetching {city.title()}, {state.upper()} {year}...")
            df = fetch_station_year(state, city, year)
            if df is not None and len(df) > 0:
                frames.append(df)
                print(f"  -> {len(df)} rows")
            time.sleep(0.5)  # be polite to the EPA server

    full = pd.concat(frames, ignore_index=True)
    full.to_csv("radnet_full.csv", index=False)
    print(f"\nSaved {len(full)} total rows across {full['LOCATION_NAME'].nunique()} "
          f"stations to radnet_full.csv")


if __name__ == "__main__":
    main()
