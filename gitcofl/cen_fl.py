from .core import FLClient
from .git_module import GitModule
from .scheduler import DefaultScheduler
import gitcofl.pytorch_connector as pytorch_connector
import uuid
import torch
import os


class CentralizedFLClient(FLClient):
    def __init__(self, repo_path, git_repo_url, git_email, access_token, model_library, model, train_loader, test_loader, val_loader, local_epochs, total_fl_rounds, interval=10):
        """
        Initializes centralized federated learning client.
        """
        super().__init__(interval=interval)
        self.repo_path = repo_path
        self.git_repo_url = git_repo_url
        self.git_email = git_email
        self.access_token = access_token
        self.model_library = model_library
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.total_fl_rounds = total_fl_rounds
        self.git = GitModule(repo_path, git_repo_url, git_email, access_token)
        self.client_id = str(uuid.uuid4())[:10]
        self.client_branch = f"client_{self.client_id}"
        self.count_fl_round = 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def pull_global_weights(self):
        """Pulls the latest global weights from the central repository."""
        print("Pulling global weights from central repository...")
        self.git.pull_global_weights(branch_name="main")

    def push_local_weights(self):
        """Pushes local weights to the central repository."""
        print("Pushing local weights to central repository...")
        self.git.push_local_weights(
            self.client_branch,
            '*',
            f'push local weight round {self.count_fl_round} from client {self.client_id}'
        )
        self.count_fl_round += 1

    def load_weights(self):
        if self.model_library == "pytorch":
            self.net, is_loaded = pytorch_connector.load_model(self.model, self.device, self.count_fl_round, self.repo_path)
            return is_loaded

    def train(self):
        """Trains the model on the local client."""
        print("Training model locally in a centralized setup...")
        if self.model_library == "pytorch":
            self.net = self.model.to(self.device)
            self.net, self.criterion = pytorch_connector.train(
                self.net,
                self.train_loader,
                self.val_loader,
                self.local_epochs,
                self.device
            )

    def test(self):
        """Tests the model on the local client."""
        print("Testing model locally in a centralized setup...")
        if self.model_library == "pytorch":
            pytorch_connector.test(self.net, self.test_loader, self.criterion, self.device)

    def save_weight(self):
        """Saves the trained model weights to the local client."""
       
        directory_path = f"{self.repo_path}/round{self.count_fl_round}"
        os.makedirs(directory_path, exist_ok=True)
        if self.model_library == "pytorch":
            pytorch_connector.save_model(self.net, directory_path, self.client_id)
        print("Saving model weight locally successfully...")

class CentralizedFLServer():
    def __init__(self, repo_path, git_repo_url, git_email, access_token, model_library, model, fl_algorithm, total_fl_rounds, number_of_client, interval=10):
        """
        Initializes centralized federated learning server.
        """
        self.repo_path = repo_path
        self.git_repo_url = git_repo_url
        self.git_email = git_email
        self.access_token = access_token
        self.git = GitModule(repo_path, git_repo_url, git_email, access_token)
        self.model_library = model_library
        self.model = model
        self.fl_algorithm = fl_algorithm
        self.total_fl_rounds = total_fl_rounds
        self.number_of_client = number_of_client
        self.count_fl_round = 1
        self.client_weights = []
        self.new_global_weight = None
        self.scheduler = DefaultScheduler(interval=interval, task=self.process)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.interval = interval

    def pull_all_client_weights(self):
        """Pulls all client weights from the central repository."""
        print('Pulling all client weights from central repository...')
        self.client_weights = []
        self.git.pull_local_weights(branch_merge_to="main")

        round_path = os.path.join(self.repo_path, f'round{self.count_fl_round}')
        try:
            weight_files = [file for file in os.listdir(round_path) if 'weight-client' in file]
            print(weight_files)
        except:
            print(f"No weight files found in round {self.count_fl_round}.")

        self.client_weights = weight_files if len(weight_files) >= self.number_of_client else None

    def aggregate(self):
        """Aggregates weights on the central server."""
        print("Aggregating all weights on the central server.")
        if self.fl_algorithm == "FedAvg":
            self.aggregate_fed_avg()

    def aggregate_fed_avg(self):
        """Aggregates weights using the FedAvg algorithm."""
        print("Aggregating weights using FedAvg algorithm...")
        if self.model_library == "pytorch":
            round_path = os.path.join(self.repo_path, f'round{self.count_fl_round}')
            self.new_global_weight = pytorch_connector.agg_fed_avg(self.client_weights, self.model, round_path)

    def push_global_weights(self):
        """Pushes the global weights to the central repository."""
        print("Pushing new global weights to the central repository...")
        global_weight_path = os.path.join(self.repo_path, f'round{self.count_fl_round}', self.new_global_weight)
        self.git.push_global_weights(
            branch_name="main",
            file='*',
            commit_message=f'push global weight round {self.count_fl_round} from server'
        )

    def process(self):
        """Processes a single round of federated learning."""
        try:
            self.pull_all_client_weights()
        except Exception as e:
            # print(f"Error pulling client weights: {e}")
            print("Cannot pulling client weights. Skipping to next schedule...")
            return

        if self.client_weights and len(self.client_weights) == self.number_of_client:
            self.aggregate()
            self.push_global_weights()
            self.count_fl_round += 1

            if self.count_fl_round > self.total_fl_rounds:
                print("Total number of rounds reached. Stopping the scheduler...")
                self.scheduler.stop()
            else:
                print(f"Round {self.count_fl_round} completed. Waiting for next round...")
        else:
            print("Not all clients have submitted their weights yet. Skipping to next schedule...")

    def start(self):
        """Starts the federated learning process with scheduling."""
        print("Starting federated learning with scheduling...")
        self.scheduler.start()
