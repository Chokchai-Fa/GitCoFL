from abc import ABC, abstractmethod
from .scheduler import DefaultScheduler

class FederatedLearning(ABC):
    def __init__(self, repo_path=None, git_repo_url=None, access_token=None, interval=10, training_function=None, scheduler=None):
        """
        Initializes federated learning with an optional custom scheduler.
        :param repo_path: Path to the Git repository
        :param git_repo_url: URL of the Git repository
        :param access_token: Access token for the Git repository
        :param interval: Time interval for checking updates (in seconds)
        :param scheduler: An instance of a scheduler, defaults to `DefaultScheduler`.
        :param training_function: The function to run for each training round.
        """
        self.repo_path = repo_path
        self.git_repo_url = git_repo_url
        self.access_token = access_token
        self.training_function = training_function or self.default_training_round
        self.scheduler = scheduler or DefaultScheduler(interval=interval, task=self.training_function)

    @abstractmethod
    def pull_global_weights(self):
        pass

    @abstractmethod
    def push_local_weights(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass

    def default_training_round(self):
        print("Performing a federated learning round...")

    def start(self):
        print("Starting federated learning with scheduling...")
        self.scheduler.start()  # Blocks execution, no threading needed
