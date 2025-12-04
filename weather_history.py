import os
import requests
import json
import argparse
from datetime import datetime, timedelta

def get_config():
    """Load configuration from the local JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not os.path.exists(config_path):
        print(f"[!] Error: config.json not found at {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_epoch(dt):
    """Helper to convert datetime to Unix timestamp."""
    return int(dt.timestamp())

def fetch_history(start_dt, end_dt, output_dir):
    # 1. Load Config
    config = get_config()
    if not config:
        return

    # 2. Extract credentials safely from JSON
    token = config.get("tempest_api_token")
    device_id = config.get("tempest_device_id")

    if not token or not device_id:
        print("[!] Error: Missing 'tempest_api_token' or 'tempest_device_id' in config.json")
        return

    # 3. Setup Chunking (Tempest API limits high-res data to 5-day blocks)
    MAX_DAYS = 5
    chunk_delta = timedelta(days=MAX_DAYS)
    
    current_start = start_dt
    all_obs = []

    print(f"--- Fetching Weather History: {start_dt} to {end_dt} ---")
    print(f"--- Device ID: {device_id} ---")

    # 4. Loop through time chunks
    while current_start < end_dt:
        current_end = min(current_start + chunk_delta, end_dt)
        
        # Standard Logic
        t_start = get_epoch(current_start)
        t_end = get_epoch(current_end)
        
        url = f"https://swd.weatherflow.com/swd/rest/observations/device/{device_id}"
        params = {
            "token": token,
            "time_start": t_start,
            "time_end": t_end,
            "format": "json"
        }

        try:
            print(f"   Requesting chunk: {current_start} -> {current_end} ...")
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            # --- THE FIX IS HERE ---
            # We now check if 'obs' exists AND if it is not None
            if data.get('obs') is not None:
                all_obs.extend(data['obs'])
            else:
                # If we get here, the API gave us a 200 OK but empty data
                pass 
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print("   [!] No data found for this period (Station might be offline or date too old).")
            else:
                print(f"   [x] API Error: {e}")
        except Exception as e:
             print(f"   [x] Unexpected Error: {e}")

        # Move to next chunk
        current_start = current_end

    # 5. Save results
    if all_obs:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"weather_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.json"
        save_path = os.path.join(output_dir, filename)
        
        final_data = {"device_id": device_id, "obs": all_obs}
        
        with open(save_path, 'w') as f:
            json.dump(final_data, f, indent=4)
        print(f"[*] Weather data saved: {len(all_obs)} records -> {save_path}")
    else:
        print("[!] No weather data retrieved for this timeline.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start time YYYYMMDDTHHMMSS")
    parser.add_argument("--end", required=True, help="End time YYYYMMDDTHHMMSS")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    try:
        s = datetime.strptime(args.start, "%Y%m%dT%H%M%S")
        e = datetime.strptime(args.end, "%Y%m%dT%H%M%S")
        fetch_history(s, e, args.out)
    except ValueError:
        print("[!] Error: Date format must be YYYYMMDDTHHMMSS (e.g., 20241029T120000)")