FROM python:3.9-slim

# Cài đặt các gói phụ thuộc hệ thống
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libpq-dev \
    libssl-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt các gói Python
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn
COPY . .

# Cổng mặc định cho ứng dụng web
EXPOSE 8080

# Lệnh khởi động mặc định
CMD ["python", "src/main.py"]
