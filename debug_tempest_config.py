import json
import requests
import os

def run_diagnostics():
    print("--- TEMPEST CONFIGURATION DIAGNOSTICS ---")

    # 1. Load the config file
    if not os.path.exists("config.json"):
        print("[!] Error: config.json not found.")
        print("    Please copy config.json.example to config.json and fill it in.")
        return

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        print("[!] Error: config.json is not valid JSON.")
        return

    # 2. Extract Configured Values
    token = config.get("tempest_api_token")
    current_station_id = str(config.get("station_id", "")).strip()
    current_device_id = str(config.get("tempest_device_id", "")).strip()

    print(f"[*] Loaded Config:")
    print(f"    - Station ID: {current_station_id}")
    print(f"    - Device ID:  {current_device_id}")
    
    if not token:
        print("[!] Error: 'tempest_api_token' is missing.")
        return

    # 3. Query the API
    print("\n[*] Connecting to WeatherFlow API...")
    url = "https://swd.weatherflow.com/swd/rest/stations"
    
    try:
        resp = requests.get(url, params={"token": token})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[!] API Connection Failed: {e}")
        print("    Check your Internet connection and API Token.")
        return

    # 4. Analyze Devices
    print("\n--- ACCOUNT HARDWARE AUDIT ---")
    
    found_configured_device = False
    config_status = "UNKNOWN"

    stations = data.get('stations', [])
    if not stations:
        print("[!] No stations found for this API Token.")
        return

    for station in stations:
        s_id = str(station['station_id'])
        s_name = station['name']
        print(f"Station: {s_name} (ID: {s_id})")

        for device in station['devices']:
            d_id = str(device['device_id'])
            d_type = device['device_type']
            d_name = device.get('device_meta', {}).get('name', 'Unnamed')
            
            # Determine Device Role
            label = ""
            is_valid_sensor = False
            
            if d_type == "ST":
                label = "  <-- [OK] VALID TEMPEST SENSOR"
                is_valid_sensor = True
            elif d_type == "SK":
                label = "  <-- [OK] VALID SKY SENSOR"
                is_valid_sensor = True
            elif d_type == "HB":
                label = "  <-- [NO] HUB (DO NOT USE)"
                is_valid_sensor = False
            elif d_type == "AR":
                label = "  <-- [OK] VALID AIR SENSOR"
                is_valid_sensor = True
            
            # Check against current config
            match_marker = ""
            if d_id == current_device_id:
                found_configured_device = True
                match_marker = " (CURRENTLY CONFIGURED)"
                
                if is_valid_sensor:
                    config_status = "GOOD"
                else:
                    config_status = "BAD_DEVICE_TYPE"

            print(f"   Device ID: {d_id} | Type: {d_type} | Name: {d_name}{label}{match_marker}")

    # 5. Final Report
    print("\n--- DIAGNOSTIC RESULTS ---")
    
    if config_status == "GOOD":
        print("[OK] CONFIGURATION LOOKS CORRECT.")
        print(f"     You are using Device {current_device_id} which is a valid sensor.")
    elif config_status == "BAD_DEVICE_TYPE":
        print("[!] CRITICAL WARNING: You have configured the HUB or an invalid device!")
        print(f"     Current ID: {current_device_id}")
        print("     Please change 'tempest_device_id' in config.json to one of the [OK] IDs above.")
    elif not found_configured_device:
        print("[!] WARNING: The configured Device ID was not found on this account.")
        print(f"     Current ID: {current_device_id}")
        print("     Please copy a valid Device ID from the list above into config.json.")
    else:
        print("[?] Status Unknown.")

if __name__ == "__main__":
    run_diagnostics()