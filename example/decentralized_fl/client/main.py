from gitcofl.decen_fl import DecentralizedFL
from dotenv import load_dotenv
import os
import time
import threading
import subprocess
import psutil
import datetime

from model import CNN, load_data_set
import torch
import uuid


FL_INTERVAL = 5
REPO_PATH = "./fl_process-"+ str(uuid.uuid4())[:3]
load_dotenv()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# torch.set_num_threads(1)

trainloader, testloader, valloader = load_data_set('./data')

# Set up network monitoring
def monitor_network():
    """Monitor network traffic for this process and write to network-stat.txt"""
    network_stats = []
    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    # Get baseline network counters for this process
    try:
        # psutil doesn't directly provide per-process network stats
        # We'll use netstat or ss command to track connections for this process
        connections = process.net_connections(kind='inet')
        # Count initial connections
        initial_connections = len(connections)
        last_connection_count = initial_connections
    except Exception as e:
        print(f"Warning: Couldn't get initial process connections: {e}")
        initial_connections = 0
        last_connection_count = 0
    
    # For tracking git processes spawned by this process
    git_pids = set()
    total_bytes_sent = 0
    total_bytes_recv = 0
    last_time = start_time
    
    with open('network-stat.txt', 'w') as f:
        f.write("# Network Statistics (Process-specific)\n")
        f.write(f"# Process ID: {os.getpid()}\n")
        f.write("# Started at {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("# Timestamp, Seconds_Elapsed, Active_Connections, TX_KB_Total, RX_KB_Total, Git_Process_Count\n")
    
    try:
        while monitoring_active:
            # Sleep for 1 second
            time.sleep(1)
            
            current_time = time.time()
            time_delta = current_time - last_time
            
            # Get all child processes, especially git processes
            try:
                children = process.children(recursive=True)
                git_process_count = sum(1 for p in children if "git" in p.name().lower())
                
                # Track active git processes
                current_git_pids = set(p.pid for p in children if "git" in p.name().lower())
                git_pids.update(current_git_pids)
                
            except Exception:
                git_process_count = 0
            
            # Get current network connections for this process
            try:
                connections = process.net_connections(kind='inet')
                connection_count = len(connections)
                
                # If connections changed, estimate data transfer
                if connection_count != last_connection_count:
                    # Estimate data transfer based on connection changes
                    # This is a rough approximation - each new connection might transfer data
                    conn_delta = abs(connection_count - last_connection_count)
                    if connection_count > last_connection_count:
                        # New connections - estimate uploads (for git push)
                        total_bytes_sent += conn_delta * 2048  # Rough estimate 2KB per new connection
                    else:
                        # Closed connections - estimate downloads (for git pull)
                        total_bytes_recv += conn_delta * 4096  # Rough estimate 4KB per closed connection
                
                last_connection_count = connection_count
                
            except Exception as e:
                connection_count = last_connection_count
            
            # Format timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            seconds_elapsed = round(current_time - start_time, 2)
            
            # Log to file
            with open('network-stat.txt', 'a') as f:
                f.write(f"{timestamp}, {seconds_elapsed}, {connection_count}, " + 
                        f"{total_bytes_sent/1024:.2f}, {total_bytes_recv/1024:.2f}, {git_process_count}\n")
            
            # Try to get network stats from git processes if any
            for git_pid in list(git_pids):
                try:
                    git_proc = psutil.Process(git_pid)
                    if git_proc.is_running() and git_proc.status() != psutil.STATUS_ZOMBIE:
                        # Git process is active, estimate some network activity
                        if "fetch" in ' '.join(git_proc.cmdline()).lower() or "pull" in ' '.join(git_proc.cmdline()).lower():
                            total_bytes_recv += 2048  # Estimate 2KB download per git fetch/pull operation
                        if "push" in ' '.join(git_proc.cmdline()).lower():
                            total_bytes_sent += 4096  # Estimate 4KB upload per git push operation
                    else:
                        git_pids.remove(git_pid)  # Remove terminated git processes
                except psutil.NoSuchProcess:
                    git_pids.remove(git_pid)  # Remove terminated git processes
                except Exception:
                    pass  # Ignore other errors
                
            last_time = current_time
                
    except Exception as e:
        with open('network-stat.txt', 'a') as f:
            f.write(f"# Error occurred: {str(e)}\n")
    finally:
        # Final log entry
        with open('network-stat.txt', 'a') as f:
            f.write("# Monitoring stopped at {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            total_time = time.time() - start_time
            f.write(f"# Total duration: {total_time:.2f} seconds\n")
            f.write(f"# Total estimated bytes sent: {total_bytes_sent/1024:.2f} KB\n")
            f.write(f"# Total estimated bytes received: {total_bytes_recv/1024:.2f} KB\n")
            f.write("# Note: These are estimated values based on process connections and Git activity\n")

# Start monitoring in a separate thread
monitoring_active = True
network_thread = threading.Thread(target=monitor_network)
network_thread.daemon = True
network_thread.start()

print("Network monitoring started. Statistics will be saved to network-stat.txt")

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
    local_epochs=10
    total_fl_rounds=5,
    sampling_no=int(os.environ.get('SAMPLE_NO')),
    interval=FL_INTERVAL)

try:
    fl.start()  # Starts the federated learning process with scheduled checks
except KeyboardInterrupt:
    print("Process interrupted by user")
except Exception as e:
    print(f"Error in FL process: {str(e)}")
finally:
    # Stop network monitoring
    monitoring_active = False
    network_thread.join(timeout=2)  # Wait for the monitoring thread to finish
    print("Network monitoring stopped. Results saved to network-stat.txt")
