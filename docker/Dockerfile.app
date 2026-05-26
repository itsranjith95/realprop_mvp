FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["streamlit", "run", "src/app/mainapp.py", "--server.port=8501", "--server.address=0.0.0.0"]