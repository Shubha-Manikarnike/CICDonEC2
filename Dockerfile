FROM python:3.7

RUN pip install numpy pandas scikit-learn pytest requests fastapi uvicorn joblib

COPY ./ /api

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0"]
