# docker build -t fschlatt/language-identification:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN curl -o code/model.joblib https://cloud.uni-jena.de/apps/files/files/96265620?dir=/nlp/download/model.joblib`

ADD run.py /code/run.py
ADD model.joblib /code/model.joblib

ENTRYPOINT [ "python3", "/code/run.py" ]