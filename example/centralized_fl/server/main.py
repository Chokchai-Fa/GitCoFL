## main.py for federated learning serverfrom gitcofl.cen_fl import CentralizedFLClient
from gitcofl.cen_fl import CentralizedFLServer
from dotenv import load_dotenv
import os
from model import CNN

FL_INTERVAL = 5
REPO_PATH = "./fl_process"
load_dotenv()

# Initialize centralized federated learning with a Git repository and scheduling
fl = CentralizedFLServer(
    repo_path=REPO_PATH, 
    git_repo_url=os.environ.get('GIT_FL_REPO'), 
    git_email=os.environ.get('GIT_EMAIL'),
    access_token=os.environ.get('GIT_ACCESS_TOKEN'),
    model_library='pytorch', 
    model=CNN(),
    fl_algorithm='FedAvg',
    total_fl_rounds=int(os.environ.get('TOTAL_FL_ROUNDS')),
    number_of_client=int(os.environ.get('NUMBER_OF_CLIENT')),
    interval=FL_INTERVAL)

fl.start()  # Starts the federated learning process with a scheduled check every 30 seconds
