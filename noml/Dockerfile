FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install necessary packages for compilation
RUN apt-get update && \
    apt-get install -y build-essential cmake gcc g++ && \
    apt-get clean

# Install necessary Python packages
RUN pip install tira sentence-transformers

ADD run.py /code/run.py
ADD bert_model /code/bert_model

ENTRYPOINT [ "python3", "/code/run.py" ]
