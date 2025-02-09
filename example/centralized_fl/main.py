from gitcofl.cen_fl import CentralizedFL

# Initialize centralized federated learning with a Git repository and scheduling
fl = CentralizedFL(repo_path="./model_repo", interval=10)
fl.start()  # Starts the federated learning process with a scheduled check every 30 seconds
