import subprocess
import sys

def run_script(script_name):
    """Run a Python script and handle errors"""
    try:
        print(f"Running {script_name}...")
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"{script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    # Run the scripts in order
    if run_script("script_generate_plot.py"):
        run_script("script_create_video.py")

if __name__ == "__main__":
    main()