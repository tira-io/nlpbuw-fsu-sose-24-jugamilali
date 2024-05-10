# docker build -t fschlatt/language-identification:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install necessary packages for compilation
RUN apt-get update && \
    apt-get install -y build-essential cmake && \
    apt-get install -y gcc g++ && \
    apt-get clean

# Install fasttext
RUN pip install fasttext

ADD run.py /code/run.py
RUN curl -o /code/model.bin https://cloud.uni-jena.de/s/wEE39bZoJKAJS2p/download/fasttext_model.bin

ENTRYPOINT [ "python3", "/code/run.py" ]