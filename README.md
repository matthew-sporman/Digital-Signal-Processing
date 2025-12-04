# digital-signal-processing
Processing SWIFT buoy hydrophone data from the Arctic Ocean.



🛠️ Installation Set-up to run this project...
1. Prerequisites
You need Miniconda or Anaconda installed.

2. Set up the Environment
Open your terminal in this folder and run the following commands to install Python, NumPy, SciPy, Matplotlib, and Requests automatically:
`conda env create -f environment.yml`
`conda activate Digital-Signal-Processing`

⚙️ Configuration:
RAW_AUDIO_DIR` = "C:\{YOUR_DIRECTORY}"
SEGMENT_DIR = "C:\{YOUR_DIRECTORY}\output_segments"
ANALYSIS_DIR = "C:\{YOUR_DIRECTORY}\output_charts"
SEGMENT_MINUTES = 12, 30, 60 (just your time in minutes!)
FS_CALIBRATION_LEVEL = 170.0 (Sensitivity of Hydrophone; Value can be changed if working with non-SWIFT buoys, but likely will not need this to change.)

This project requires a Tempest API token to fetch weather data.
Rename config.json.example to config.json.

Open it and fill in your credentials:

{
    "tempest_api_token": "YOUR_LONG_TOKEN_STRING",
    "station_id": "82486",
    "tempest_device_id": "YOUR_DEVICE_ID" 
}

NOTE: station_id: 82486 refers to the GLRC station in the Portage Canal.

After setting these to use this program, you simply run the file: `run_pipeline.py`
The execution progress can be traced via the output in the Terminal I/O.


Project Structure:
* [ ] `run_pipeline.py` (Main script)
* [ ] `wav_split.py` (WAV file splitter)
* [ ] `weather_history.py` (TEMPEST API file)
* [ ] `snap_hydrophone_processing.py` (File processing)
* [ ] `debug_tempest_config.py` (The diagnostic tool if using different Tempest stations to find correct Device ID, which is distinct from a Station ID)
* [ ] `environment.yml` (For installation)
* [ ] `README.md` (The instructions above)
* [ ] `config.json.example` (Dummy keys -> create your own config.json with this structure)
* [ ] `.gitignore`
* [ ] `config.json` (⚠️ **DELETE THIS or ADD TO GITIGNORE before sharing! 🛑 Never upload your API keys please it's really bad practice. 🛑** ⚠️)

TO REITERATE, `tempest_api_token` should NEVER be uploaded to GitHub or any online accessible source to prevent abuse.

🔧 Troubleshooting:
"No Data Found" for Weather? Run the diagnostic tool: `debug_tempest_config.py`
This will verify if your API token is working and if you have selected the correct Device ID (Sensor vs Hub).