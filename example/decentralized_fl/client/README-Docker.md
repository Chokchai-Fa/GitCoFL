# GitCoFL Decentralized Client Docker Setup

This Docker setup allows you to run the GitCoFL decentralized federated learning client in a containerized environment.

## Quick Start

### 1. Build the Docker Image

From the project root directory:

```bash
cd /home/chokchai-fa/cs_chula/thesis/GitCoFL
docker build -f example/decentralized_fl/client/Dockerfile -t gitcofl-decentralized-client .
```

Or use the provided build script:

```bash
cd example/decentralized_fl/client
./build-docker.sh
```

### 2. Setup Environment Variables

Copy and edit the environment file:

```bash
cp .env.example .env
# Edit .env with your actual values
```

### 3. Run the Container

Run a single client:

```bash
docker run --env-file .env -v $(pwd)/data:/app/client/data gitcofl-decentralized-client
```

## Multi-Client Setup

To run multiple clients for decentralized FL, create multiple environment files:

```bash
# Client 1
cp .env.example .env.client1
# Edit with SAMPLE_NO=1

# Client 2  
cp .env.example .env.client2
# Edit with SAMPLE_NO=2

# Run multiple clients
docker run --name fl-client1 --env-file .env.client1 -d gitcofl-decentralized-client
docker run --name fl-client2 --env-file .env.client2 -d gitcofl-decentralized-client
```

## Environment Variables

- `GIT_FL_REPO`: Git repository URL for federated learning coordination
- `GIT_ACCESS_TOKEN`: GitHub access token for repository access
- `GIT_EMAIL`: Git email for commits
- `SAMPLE_NO`: Client identifier (unique number for each client)

## Volume Mounts

- `/app/client/data`: Mount your local data directory to persist training data

## Viewing Logs

```bash
# View logs for a running container
docker logs fl-client1

# Follow logs in real-time
docker logs -f fl-client1
```

## Stopping Containers

```bash
# Stop a specific client
docker stop fl-client1

# Stop all running GitCoFL clients
docker stop $(docker ps -q --filter ancestor=gitcofl-decentralized-client)
```
