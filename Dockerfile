FROM python:3.8

RUN pip install numpy pandas scikit-learn pytest requests fastapi uvicorn joblib pyyaml

COPY ./ /api

ENV PYTHONPATH=/api

WORKDIR /api

CMD CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
