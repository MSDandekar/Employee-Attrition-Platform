FROM python:3.11-slim

WORKDIR /app

COPY deployment/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY deployment/app.py deployment/app.py

COPY output.1/best_attrition_model.pkl output.1/best_attrition_model.pkl

EXPOSE 8000

CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]