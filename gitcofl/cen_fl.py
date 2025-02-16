from .core import FederatedLearning
from .git_module import GitModule

class CentralizedFL(FederatedLearning):
    def __init__(self, repo_path, git_repo_url, access_token, interval=10, training_function=None, test_function=None):
        """
        Initializes centralized federated learning.
        :param repo_path: Path to the Git repository for the centralized model.
        :param git_repo_url: URL of the Git repository.
        :param access_token: Access token for the Git repository.
        :param interval: Time interval to check for global weights.
        :param training_function: The function to run for each training round.
        """
        super().__init__(interval=interval)
        self.repo_path = repo_path
        self.git_repo_url = git_repo_url
        self.access_token = access_token
        self.git = GitModule(repo_path, git_repo_url, access_token)

        self.training_function = training_function
        self.test_function = test_function

    def pull_global_weights(self, client_branch: str):
        """Pulls the latest global weights from the central repository."""
        print("Pulling global weights from central repository...")
        self.git.pull_global_weights(branch_name=client_branch)

    def push_local_weights(self):
        """Pushes local weights to the central repository."""
        print("Pushing local weights to central repository...")
        self.git.push()

    def train(self):
        """Trains the model on the local client."""
        print("Training model locally in a centralized setup...")

    def aggregate(self):
        """Aggregates weights on the central server."""
        print("Aggregating weights on the central server.")
