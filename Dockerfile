FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --retries 10 -r requirements.txt

COPY src/ src/

# Preprocess data and train the model inside the container
ENV MLFLOW_ALLOW_FILE_STORE="true"
RUN python src/preprocess.py && python src/train.py

EXPOSE 8000

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]