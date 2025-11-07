# digital-signal-processing
Processing SWIFT buoy hydrophone data from the Artic Ocean.



Firstly, before running the following variables will need be updated:

RAW_AUDIO_DIR
SEGMENT_DIR
ANALYSIS_DIR
SEGMENT_MINUTES
FS_CALIBRATION_LEVEL


After setting these to use this program, you simply run the file: run_pipeline.py

Follow output trace to detect errors if any occur!



RECOMMENDED SET-UP:

RAW_AUDIO_DIR = "C:\{YOUR_DIRECTORY}"
SEGMENT_DIR = "C:\{YOUR_DIRECTORY}\output_segments"
ANALYSIS_DIR = "C:\{YOUR_DIRECTORY}\output_charts"

SEGMENT_MINUTES = 12, 30, 60 (just your time in minutes!)
FS_CALIBRATION_LEVEL = 170.0 (Sensitivity of Hydrophone; Value can be changed if working with non-SWIFT bouys, but likely will not need this to change.)