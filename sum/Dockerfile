# docker build -t fschlatt/authorship-verification-naive-bayes:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install necessary packages for compilation
RUN apt-get update && \
    apt-get install -y build-essential cmake gcc g++ && \
    apt-get clean

# Install necessary Python packages
RUN pip install tira transformers torch accelerate -U

ADD run.py /code/run.py
RUN curl -o /code/summarizer_model/generation_config.json https://cloud.uni-jena.de/s/WNmXbq5HgfoM7dg/download/generation_config.json
RUN curl -o /code/summarizer_model/model.safetensors https://cloud.uni-jena.de/s/jncgNY3GNnppMfM/download/model.safetensors
RUN curl -o /code/summarizer_model/config.json https://cloud.uni-jena.de/s/WNmXbq5HgfoM7dg/download/config.json

ENTRYPOINT [ "python3", "/code/run.py" ]