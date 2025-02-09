from gitcofl.cen_fl import CentralizedFL
from dotenv import load_dotenv
import os


FL_INTERVAL = 10 
REPO_PATH = "./fl_process"
load_dotenv()

# Initialize centralized federated learning with a Git repository and scheduling
fl = CentralizedFL(
    repo_path=REPO_PATH, 
    git_repo_url=os.environ.get('GIT_FL_REPO'), 
    access_token=os.environ.get('GIT_ACCESS_TOKEN'), 
    interval=FL_INTERVAL)

fl.start()  # Starts the federated learning process with a scheduled check every 30 seconds
