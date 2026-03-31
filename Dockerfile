FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Run training + evaluation inside the container
RUN python src/train.py
RUN python src/evaluate.py

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard.py", "--server.address=0.0.0.0"]