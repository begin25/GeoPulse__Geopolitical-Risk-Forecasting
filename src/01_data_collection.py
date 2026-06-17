import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import functools

# Project Constants
MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
DAYS_TO_FETCH = 180 
TARGET_COUNTRIES = ['ISR', 'IRN', 'RUS', 'UKR', 'IND', 'PAK', 'CHN', 'TWN']

USECOLS = [1, 7, 17, 30, 31, 32, 34]
GDELT_NAMES = ['SQLDATE', 'Actor1CountryCode', 'Actor2CountryCode', 'QuadClass', 'GoldsteinScale', 'NumMentions', 'AvgTone']

def ensure_dirs():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def get_file_urls(days_back=7):
    print("Fetching GDELT master file list...")
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d")
    res = requests.get(MASTER_URL)
    res.raise_for_status()

    filtered_urls = []
    # Strict boundary check to prevent decade-long downloads
    for line in res.text.splitlines():
        if "export.CSV.zip" in line:
            url = line.split()[-1]
            filename = url.split("/")[-1]
            date_str = filename[:8]
            
            if date_str.isdigit() and date_str >= cutoff:
                filtered_urls.append(url)

    print(f"Found {len(filtered_urls)} files for the past {days_back} days.")
    return filtered_urls, cutoff

def download_and_filter(url, cutoff):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
            
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, sep="\t", header=None, usecols=USECOLS, names=GDELT_NAMES, low_memory=False)
                df = df.dropna(subset=['Actor1CountryCode', 'Actor2CountryCode'])

                mask = df['Actor1CountryCode'].isin(TARGET_COUNTRIES) & df['Actor2CountryCode'].isin(TARGET_COUNTRIES)
                df_filtered = df[mask].copy()

                df_filtered['SQLDATE'] = df_filtered['SQLDATE'].astype(str)
                df_filtered = df_filtered[df_filtered['SQLDATE'] >= cutoff]

                return df_filtered
    except Exception:
        return pd.DataFrame()

def main():
    ensure_dirs()
    urls, cutoff = get_file_urls(DAYS_TO_FETCH)

    print("Downloading and processing files in parallel (this may take a while)...")
    all_data = []

    bound_download = functools.partial(download_and_filter, cutoff=cutoff)

    # Parallel I/O network fetching
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(bound_download, urls)
        for i, df in enumerate(results):
            if not df.empty:
                all_data.append(df)
            if i > 0 and i % 100 == 0:
                print(f"Processed {i}/{len(urls)} files...")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        out_path = 'data/raw/conflict_events_raw.csv'
        final_df.to_csv(out_path, index=False)
        print(f"Data collection complete! Saved {len(final_df)} events to {out_path}")
    else:
        print("No relevant events found in this timeframe.")

if __name__ == "__main__":
    main()