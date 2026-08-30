# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install with high timeout and retries to survive connection drops
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --retries 10 -r requirements.txt

# Copy the API code and the trained model
COPY src/predict.py src/predict.py
COPY models/model.joblib models/model.joblib

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the server
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]