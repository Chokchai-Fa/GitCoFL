FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the GitCoFL package and setup files from project root
COPY gitcofl/ /app/gitcofl/
COPY setup.py /app/
COPY requirements.txt /app/
COPY README.md /app/

# Copy client-specific files
COPY example/decentralized_fl/client/ /app/client/
WORKDIR /app/client

# Install the gitcofl package in development mode
RUN pip install -e /app

# Install client-specific dependencies
RUN pip install -r requirements.txt

# Create data directory
RUN mkdir -p data

# Run the client
CMD ["python", "main.py"]