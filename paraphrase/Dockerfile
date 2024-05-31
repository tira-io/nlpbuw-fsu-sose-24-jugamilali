# docker build -t fschlatt/paraphrase-identification:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install necessary packages for compilation
RUN apt-get update && \
    apt-get install -y build-essential cmake && \
    apt-get install -y gcc g++ && \
    apt-get clean

RUN pip install tira \
    pip install transformers \
    pip install datasets \
    pip install accelerate -U

ADD run.py /code/run.py
RUN curl -o /code/model.safetensors https://cloud.uni-jena.de/s/8pAxQB57bsLaajw/download/model.safetensors

ENTRYPOINT [ "python3", "/code/run.py" ]