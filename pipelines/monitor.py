import psutil
import subprocess
import time
import os


def log_usage(process):
    with open("usage_log.txt", "a") as log_file:
        while process.is_running():
            try:
                cpu_usage = process.cpu_percent(interval=1)
                memory_info = process.memory_info()
                memory_usage = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
                log_file.write(
                    f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage} MB\n"
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                break


if __name__ == "__main__":
    script_path = "multilegal_pipeline.py"  # Replace with your script's name

    # Start the script
    process = subprocess.Popen(
        ["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Get the process object
    p = psutil.Process(process.pid)

    # Log the usage
    log_usage(p)

    # Wait for the process to finish
    process.wait()

    # Log completion
    with open("usage_log.txt", "a") as log_file:
        log_file.write("Script execution completed.\n")
