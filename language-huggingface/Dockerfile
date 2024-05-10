# docker build -t fschlatt/language-identification:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
RUN curl -o /code/model.joblib https://cloud.uni-jena.de/s/wEE39bZoJKAJS2p/download/fasttext_model.bin

ENTRYPOINT [ "python3", "/code/run.py" ]