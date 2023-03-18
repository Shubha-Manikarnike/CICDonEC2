FROM python:3.8

RUN pip install numpy pandas scikit-learn pytest requests fastapi uvicorn joblib pyyaml

COPY ./ /api

ENV PYTHONPATH=/api

WORKDIR /api

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
