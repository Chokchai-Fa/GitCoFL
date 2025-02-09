import git
import os

class GitModule:
    def __init__(self, repo_path: str, git_repo_url: str, access_token: str):
        """
        Initializes the GitCommunicate class for interacting with Git repositories.
        :param repo_path: Path to the local repository.
        :param git_repo_url: URL to the remote Git repository.
        :param access_token: Optional Git access token for authentication (if using HTTPS).
        """
        self.repo_path = repo_path
        self.git_repo_url = git_repo_url
        self.access_token = access_token
        self.repo = self._get_or_clone_repo()

    # def _get_or_clone_repo(self):
    #     """Clones the repository if it doesn't exist locally."""
    #     try:
    #         repo = git.Repo(self.repo_path)
    #         print(f"Repository found at {self.repo_path}")
    #     except git.exc.InvalidGitRepositoryError:
    #         print(f"Cloning repository from GitHub to {self.repo_path}...")
    #         # repo_url = f"https://github.com/{self.repo_path}"
    #         repo_url = f"https://{self.access_token}@{git_repo_url}" if self.access_token else git_repo_url
    #         repo = self.git_clone(repo_url)
    #     return repo

    # def git_clone(self, git_repo_url: str):
    #     """Clones the repository from the given URL."""
    #     repo_url = f"https://{self.access_token}@{git_repo_url}" if self.access_token else git_repo_url
    #     repo = git.Repo.clone_from(repo_url, self.repo_path)
    #     repo.git.checkout('main')  # Ensure we are on the main branch
    #     return repo

    def _get_or_clone_repo(self):
        """
        Gets existing repo or clones a new one if it doesn't exist.
        
        Returns:
            git.Repo: Git repository object
        """
        try:
            # Try to get existing repo
            if os.path.exists(self.repo_path):
                return git.Repo(self.repo_path)
            
            # If path doesn't exist and we have a URL, clone it
            if self.git_repo_url:
                print(f"Cloning repository from {self.git_repo_url} to {self.repo_path}...")
                os.makedirs(self.repo_path, exist_ok=True)
                
                # Construct URL with access token if provided
                clone_url = (f"https://{self.access_token}@{self.git_repo_url}" 
                           if self.access_token else self.git_repo_url)
                
                return git.Repo.clone_from(clone_url, self.repo_path)
            
            # If no URL provided, initialize new repo
            print(f"Initializing new repository at {self.repo_path}...")
            os.makedirs(self.repo_path, exist_ok=True)
            return git.Repo.init(self.repo_path)
            
        except git.exc.GitCommandError as e:
            print(f"Git command error: {e}")
            raise
        except Exception as e:
            print(f"Error initializing repository: {e}")
            raise

    def push_local_weights(self, branch_name: str, file: str, commit_message: str, email: str):
        """Pushes the local weights to the Git repository."""
        self._checkout_branch(branch_name)
        self.repo.config_writer().set_value("user", "email", email).release()
        self.repo.config_writer().set_value("user", "name", "Chokchai-Fa").release()
        
        self.repo.git.add(file)
        self.repo.git.commit('-m', commit_message)
        self.repo.git.push('origin', branch_name)
        print("Push file successfully")

    def pull_global_weights(self, branch_name: str):
        """Pulls the global weights from the Git repository."""
        self.repo.remotes.origin.fetch()
        self.repo.git.checkout(branch_name)
        self.repo.git.merge('origin/main')
        print("Pull file successfully")

    def _checkout_branch(self, branch_name: str):
        """Checks out to the given branch, creating it if necessary."""
        if branch_name not in self.repo.git.branch():
            self.repo.git.branch(branch_name)
        self.repo.git.checkout(branch_name)
