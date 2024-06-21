FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install necessary packages for compilation and unzip
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake gcc g++ unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary Python packages
RUN pip install --no-cache-dir tira transformers torch accelerate -U

# Add the script to the container
ADD run.py /code/run.py

# Download and unzip the summarizer model
RUN curl -L -o /code/sumbart.zip https://cloud.uni-jena.de/s/R3qFYxJ9Ln2GjKM/download && \
    unzip /code/sumbart.zip -d /code/ && \
    rm /code/sumbart.zip

# Set the entrypoint
ENTRYPOINT ["python3", "/code/run.py"]
