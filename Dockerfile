# 使用 Python 官方镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有项目文件到容器中
COPY . .

# 安装 Flask
RUN pip install flask

# 复制静态文件
COPY static /app/static

# 暴露端口（如果你的应用需要的话，端口号可以根据实际情况修改）
EXPOSE 8000

# 修改启动命令
CMD ["python", "server.py"] 