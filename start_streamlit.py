
import subprocess
import sys
import os

os.chdir("liftos-streamlit")
subprocess.run([
    sys.executable, "-m", "streamlit", "run", "Home.py",
    "--server.port", "8501",
    "--server.address", "0.0.0.0",
    "--browser.gatherUsageStats", "false"
])
