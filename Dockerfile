FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY . .

# 数据目录
RUN mkdir -p data/handbooks data/seed_cases data/chroma

EXPOSE 8000 8501

CMD ["uvicorn", "api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
