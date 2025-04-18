from .core import FLClient
from .git_module import GitModule
import gitcofl.pytorch_connector as pytorch_connector
import torch
import uuid
import os
import random

class DecentralizedFL(FLClient):
    def __init__(self, repo_path, git_repo_url, git_email, access_token, model_library, model, fl_algorithm, train_loader, test_loader, val_loader, local_epochs, total_fl_rounds, sampling_no, interval=10):
        """
        Initializes decentralized federated learning.
        """
        super().__init__(interval=interval)
        self.repo_path = repo_path
        self.git_repo_url = git_repo_url
        self.git_email = git_email
        self.access_token = access_token
        self.model_library = model_library
        self.model = model
        self.fl_algorithm = fl_algorithm
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.total_fl_rounds = total_fl_rounds
        self.git = GitModule(repo_path, git_repo_url, git_email, access_token)
        self.client_id = str(uuid.uuid4())[:10]
        self.client_branch = f"client_{self.client_id}"
        self.count_fl_round = 1
        self.weights_to_merge_round = 1 # Number of rounds of weights to merge (ex. for round2, merge from round1)
        self.client_weights = []
        self.new_global_weight = None
        self.sampling_no = sampling_no
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def validate_parameter(self):
        pass

    def pull_global_weights(self):
        """Pulls global weights from peer repositories."""
        print('Pulling all client weights from central repository...')
        self.client_weights = []
        self.git.pull_local_weights(branch_merge_to="main")

        round_path = os.path.join(self.repo_path, f'round{self.weights_to_merge_round}')
        try:
            weight_files = [file for file in os.listdir(round_path) if 'weight-client' in file]
            print(weight_files)
        except:
            print(f"No weight files found in round {self.count_fl_round}.")

        self.client_weights = weight_files if len(weight_files) >= self.sampling_no else None
    
    def after_pull_global_weights(self):
        """After pulling global weights, aggregate the model weights to the local repository."""
        self.aggregate()

    def push_local_weights(self):
        """Pushes local weights to peer repositories."""
        self.git.push_local_weights(
            self.client_branch,
            '*',
            f'push local weight round {self.count_fl_round} from client {self.client_id}'
        )
        self.count_fl_round += 1
        self.weights_to_merge_round = self.count_fl_round - 1

    def load_weights(self):
        model_path = os.path.join(self.repo_path, f'round{self.weights_to_merge_round}')
        if self.model_library == "pytorch":
            self.net, is_loaded = pytorch_connector.load_model(self.model, self.device, self.count_fl_round, model_path, 'decen')
            return is_loaded
        else:
            self.raiseUnknownModelLibrary()

    def train(self):
        """Trains the model on the local client in a decentralized setup."""
        print("Training model locally in a decentralized setup...")
        if self.model_library == "pytorch":
            self.net = self.model.to(self.device)
            self.net, self.criterion = pytorch_connector.train(
                self.net,
                self.train_loader,
                self.val_loader,
                self.local_epochs,
                self.device
            )
        else:
            self.raiseUnknownModelLibrary()

    def test(self):
        """Tests the model on the local client in a decentralized setup."""
        print("Testing model locally in a decentralized setup...")
        if self.model_library == "pytorch":
            pytorch_connector.test(self.net, self.test_loader, self.criterion, self.device)
        else:
             self.raiseUnknownModelLibrary()

    def save_weight(self):
        """Saves the model weights to the local repository."""
        print("Saving model weights to local repository...")
        directory_path = f"{self.repo_path}/round{self.count_fl_round}"
        os.makedirs(directory_path, exist_ok=True)
        if self.model_library == "pytorch":
            pytorch_connector.save_model(self.net, directory_path, self.client_id)
            print("Saving model weight locally successfully...")
        else:
            self.raiseUnknownModelLibrary()

    def aggregate(self):
        """Aggregates weights on the central server."""
        print("Performing decentralized aggregation with peers.")
        if self.fl_algorithm == "FedAvg":
            self.aggregate_fed_avg()
        else:
            self.raiseUnknownFLAlgorithm()

    def aggregate_fed_avg(self):
        """Performs decentralized aggregation with peers."""
        print("Performing decentralized aggregation by fed avg algorithm...")
        sampled_weights = random.sample(self.client_weights, int(self.sampling_no))
        if self.model_library == "pytorch":
            round_path = os.path.join(self.repo_path, f'round{self.weights_to_merge_round}')
            self.new_global_weight = pytorch_connector.agg_fed_avg(sampled_weights, self.model, round_path, 'decen')
        else:
            self.raiseUnknownModelLibrary()
        
    def raiseUnknownModelLibrary(self):
        print(f"Unknown model library: {self.model_library}.")
        raise ValueError(f"Unknown model library: {self.model_library}.")
    
    def raiseUnknownFLAlgorithm(self):
        print(f"Unknown FL algorithm: {self.fl_algorithm}.")
        raise ValueError(f"Unknown FL algorithm: {self.fl_algorithm}.")
