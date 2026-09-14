FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY outputs/model/model.pkl ./model/model.pkl
ENV MODEL_PATH=/app/model/model.pkl
ENV DECISION_THRESHOLD=0.5
EXPOSE 8000
CMD ["uvicorn", "src.serve.main:app", \
"--host", "0.0.0.0", "--port", "8000"]