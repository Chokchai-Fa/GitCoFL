import git
import os

class GitModule:
    def __init__(self, repo_path: str, git_repo_url: str, git_email: str, access_token: str):
        """
        Initializes the GitCommunicate class for interacting with Git repositories.
        :param repo_path: Path to the local repository.
        :param git_repo_url: URL to the remote Git repository.
        :param access_token: Optional Git access token for authentication (if using HTTPS).
        """
        self.repo_path = repo_path
        self.git_repo_url = git_repo_url
        self.git_email = git_email
        self.access_token = access_token
        self.repo = self._get_or_clone_repo()

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
                repo = git.Repo.clone_from(clone_url, self.repo_path)
                repo.git.checkout('main')
                return repo
            
            # print(f"Initializing new repository at {self.repo_path}...")
            # os.makedirs(self.repo_path, exist_ok=True)
            # return git.Repo.init(self.repo_path)
            
        except git.exc.GitCommandError as e:
            print(f"Git command error: {e}")
            raise
        except Exception as e:
            print(f"Error initializing repository: {e}")
            raise

    def push_local_weights(self, branch_name: str, file: str, commit_message: str):
        """Pushes the local weights to the Git repository."""
        self._checkout_branch(branch_name)
        self.repo.config_writer().set_value("user", "email", self.git_email).release()
        self.repo.config_writer().set_value("user", "name", "fl-client").release()
        
        self.repo.git.add(file)
        self.repo.git.commit('-m', commit_message)
        self.repo.git.push('origin', branch_name)
        print("Push file successfully")


    def push_global_weights(self, branch_name: str, file: str, commit_message: str):
        """Pushes the global weights to the Git repository."""
        self._checkout_branch(branch_name)
        self.repo.config_writer().set_value("user", "email", self.git_email).release()
        self.repo.config_writer().set_value("user", "name", "fl-server").release()

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

    # This method is used to pull local weights from the Git repository, using by centralliezed FL server and decentralized FL client
    def pull_local_weights(self, branch_merge_to: str):
        """Pulls the local weights from the Git repository."""
        self.repo.config_writer().set_value("user", "email", self.git_email).release()
        self.repo.config_writer().set_value("user", "name", "fl-server").release()

        self.repo.remotes.origin.fetch()

        # Check out to branch that want to merge weights from local branchs
        self.repo.git.checkout(branch_merge_to)


        # List all remote branches, including those that haven't been checked out locally
        remote_branches = [ref.name.split('/')[-1] for ref in self.repo.remotes.origin.refs]
        print("Remote branches:", remote_branches)

        for b in remote_branches:
            if b != branch_merge_to:  # Avoid merging with the current branch itself
                try:
                    self.repo.git.merge(f'origin/{b}')  # Merge the remote branch into the current branch
                except git.exc.GitCommandError as e:
                    print(f"Merge conflict or issue with {b}: {e}")

        # Push to the remote, set the upstream branch if it doesn't exist
        origin = self.repo.remote(name='origin')
        try:
            origin.push()  # Attempt to push
        except git.exc.GitCommandError as e:
            if "has no upstream branch" in str(e):
                # Set the upstream for the branch and push again
                self.repo.git.push('--set-upstream', 'origin', branch_merge_to)
                origin.push()  # Push after setting upstream
            else:
                raise e