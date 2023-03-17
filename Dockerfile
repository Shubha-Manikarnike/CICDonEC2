FROM python:3.7

RUN pip install -r requirements.txt

COPY ./ /api

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0"]