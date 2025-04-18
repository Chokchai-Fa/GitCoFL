from gitcofl.decen_fl import DecentralizedFL
from dotenv import load_dotenv
import os

from model import CNN, load_data_set
import torch
import uuid


FL_INTERVAL = 5
REPO_PATH = "./fl_process-"+ str(uuid.uuid4())[:3]
load_dotenv()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

trainloader, testloader, valloader = load_data_set('./data')

# Initialize centralized federated learning with a Git repository and scheduling
fl = DecentralizedFL(
    repo_path=REPO_PATH, 
    git_repo_url=os.environ.get('GIT_FL_REPO'), 
    access_token=os.environ.get('GIT_ACCESS_TOKEN'),
    git_email=os.environ.get('GIT_EMAIL'),
    model_library='pytorch', 
    model=CNN(),
    fl_algorithm='FedAvg',
    train_loader=trainloader,
    test_loader=valloader,
    val_loader=valloader,
    local_epochs=10,
    total_fl_rounds=3,
    sampling_no=int(os.environ.get('SAMPLE_NO')),
    interval=FL_INTERVAL)

fl.start()  # Starts the federated learning process with a scheduled check every 30 seconds
