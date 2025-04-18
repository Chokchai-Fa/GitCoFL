from .core import FederatedLearning
from .git_module import GitModule

class DecentralizedFL(FederatedLearning):
    def __init__(self, repo_path, peers,access_token , model_library, interval=10):
        """
        Initializes decentralized federated learning.
        :param repo_path: Path to the Git repository for this local model.
        :param peers: List of peer repositories to sync weights with.
        :param interval: Time interval to check for global weights.
        :param access_token: Optional access token for GitHub authentication.
        """
        super().__init__(interval=interval)
        self.repo_path = repo_path
        self.peers = peers  # List of peer repositories
        self.model_library = model_library
        self.git = GitModule(repo_path, access_token)

    def pull_global_weights(self):
        """Pulls global weights from peer repositories."""

        ## not implement yet

        # print("Pulling global weights from peers...")
        # for peer in self.peers:
        #     self.git.pull_global_weights(peer)

    def push_local_weights(self):
        """Pushes local weights to peer repositories."""
        
        ## not implement yet
        
        # print("Pushing local weights to peers...")
        # self.git.push_local_weights(branch_name="main", file="weights.json", commit_message="Update local weights", email="example@example.com")

    def train(self):
        """Trains the model on the local client in a decentralized setup."""
        print("Training model locally in a decentralized setup...")

    def test(self):
        """Tests the model on the local client in a decentralized setup."""
        print("Testing model locally in a decentralized setup...")

    def aggregate(self):
        """Performs decentralized aggregation with peers."""
        print("Performing decentralized aggregation with peers.")