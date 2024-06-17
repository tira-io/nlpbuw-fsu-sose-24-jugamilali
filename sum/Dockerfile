# Base image
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install necessary packages for compilation and unzip
RUN apt-get update && \
    apt-get install -y build-essential cmake gcc g++ unzip && \
    apt-get clean

# Install necessary Python packages
RUN pip install tira transformers torch accelerate -U

# Add the script to the container
ADD run.py /code/run.py

# Download and unzip the summarizer model
RUN curl -o /code/summarizer_model.zip https://cloud.uni-jena.de/s/rRZjmJHjJjYACFB/download && \
    unzip /code/summarizer_model.zip -d /code/

RUN curl -o /code/summarizer_tokenizer.zip https://cloud.uni-jena.de/s/WSDsickiEcSzTNo/download && \
    unzip /code/summarizer_tokenizer.zip -d /code/

# Set the entrypoint
ENTRYPOINT [ "python3", "/code/run.py" ]
