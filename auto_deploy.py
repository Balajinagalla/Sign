import subprocess
import time
import os
import sys

def kill_old_processes():
    print("🧹 Cleaning up old connections and server processes...")
    if sys.platform == "win32":
        # Forcefully kill any python or ssh processes related to streamlit/tunnels to clear port 8501
        subprocess.run(['powershell', '-Command', 'Stop-Process -Name "ssh" -Force -ErrorAction SilentlyContinue'], capture_output=True)
        # We don't kill all python because the user might have others, but we try to kill streamlit ones
        subprocess.run(['powershell', '-Command', 'Get-Process | Where-Object { $_.CommandLine -like "*streamlit*" } | Stop-Process -Force -ErrorAction SilentlyContinue'], capture_output=True)
    time.sleep(1)

def run_auto_deploy():
    kill_old_processes()
    
    print("="*60)
    print("🚀 SIGN TRANSLATOR - BOOTING CLEAN SESSION")
    print("="*60)
    
    # 1. Start Streamlit
    print(f"📡 Starting Streamlit (Port 8501)...")
    log_file = open("streamlit_log.txt", "w")
    streamlit_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.headless", "true"],
        stdout=log_file,
        stderr=log_file,
        shell=True
    )
    
    time.sleep(3)
    
    # 2. Launch Serveo Tunnel with keep-alive
    print(f"🔗 Opening Secure ZERO-PASSWORD Tunnel...")
    print("\n" + "⭐"*20)
    print("✨ YOUR APP IS NOW LIVE ON THE WEB!")
    print("⭐"*20)
    
    # Added ServerAliveInterval to keep the connection stable
    ssh_cmd = 'ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -R 80:localhost:8501 serveo.net'
    
    try:
        os.system(ssh_cmd)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        streamlit_proc.terminate()

if __name__ == "__main__":
    run_auto_deploy()
