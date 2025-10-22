import time
import os
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import sys

# Define the directories
JOB_DIR_TODO = os.path.join(os.path.dirname(__file__), "jobs", "todo")
JOB_DIR_DONE = os.path.join(os.path.dirname(__file__), "jobs", "done")

class JobHandler(FileSystemEventHandler):
    """Handles new job files created in the 'todo' directory."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            print(f"[JobConsumer] Detected new job file: {os.path.basename(event.src_path)}")
            self.process_job(event.src_path)

    def process_job(self, job_path):
        try:
            with open(job_path, 'r') as f:
                job_data = json.load(f)
            
            print(f"[JobConsumer] Processing job: {job_data.get('job_type')} for asset {job_data.get('asset')}")

            if job_data.get('job_type') == 'RECYCLE_DNA':
                # For now, we will just re-run the main gp_framework.
                # A more advanced implementation would pass the 'seed_dna_from'
                # parameter to the GP framework to influence the initial population.
                
                # We run the main Forge cycle in a separate process.
                # This assumes gp_framework.py is executable and in the same directory.
                script_path = os.path.join(os.path.dirname(__file__), "gp_framework.py")
                
                # Using the same Python interpreter that is running this script
                process = subprocess.Popen([sys.executable, script_path])
                
                print(f"[JobConsumer] Launched new Forge cycle (PID: {process.pid}) to generate replacement strategy.")
                
            else:
                print(f"[JobConsumer] WARNING: Unknown job type '{job_data.get('job_type')}'")

            # Move the processed job file to the 'done' directory
            done_path = os.path.join(JOB_DIR_DONE, os.path.basename(job_path))
            os.rename(job_path, done_path)
            print(f"[JobConsumer] Job completed and moved to '{done_path}'")

        except Exception as e:
            print(f"[JobConsumer] ERROR processing job {job_path}: {e}")
            # Optionally, move to an 'error' directory
            # error_path = os.path.join(JOB_DIR_ERROR, os.path.basename(job_path))
            # os.rename(job_path, error_path)


def start_job_consumer():
    """Starts the watchdog observer to monitor the job directory."""
    # Ensure directories exist
    os.makedirs(JOB_DIR_TODO, exist_ok=True)
    os.makedirs(JOB_DIR_DONE, exist_ok=True)
    
    event_handler = JobHandler()
    observer = Observer()
    observer.schedule(event_handler, JOB_DIR_TODO, recursive=False)
    
    print(f"[JobConsumer] Watching for new jobs in: {JOB_DIR_TODO}")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_job_consumer()
