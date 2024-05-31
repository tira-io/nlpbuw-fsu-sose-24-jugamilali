# docker build -t fschlatt/paraphrase-identification:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install necessary packages for compilation
RUN apt-get update && \
    apt-get install -y build-essential cmake gcc g++ && \
    apt-get clean

# Install necessary Python packages
RUN pip install tira transformers datasets accelerate -U

# Create the /code directory and add the Python script and model
ADD run.py /code/run.py
ADD config.json /code/config.json
ADD special_tokens_map.json /code/special_tokens_map.json
ADD tokenizer_config.json /code/tokenizer_config.json
ADD vocab.txt /code/vocab.txt
RUN curl -o /code/model.safetensors https://cloud.uni-jena.de/s/8pAxQB57bsLaajw/download/model.safetensors

ENTRYPOINT [ "python3", "/code/run.py" ]
