# docker build -t fschlatt/language-identification:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

<<<<<<< HEAD
RUN curl -o language-identificator/model.joblib https://cloud.uni-jena.de/apps/files/files/96265620?dir=/nlp/model.joblib`

=======
>>>>>>> a2daf3776a3fbf72a6f0b53259bff5d19f1711af
ADD run.py /code/run.py
ADD model.joblib /code/model.joblib

ENTRYPOINT [ "python3", "/code/run.py" ]