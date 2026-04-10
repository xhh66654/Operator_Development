#1.选择基础环境镜像
FROM python:3.12-slim

#2.设置环境变量，确保日志打印
ENV PYTHONUNBUFFERED=1

#3.设置工作目录
WORKDIR /app

#4.依赖文件
COPY requirements.txt .

#5.安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

#6.复制工作目录
COPY . .

#7.对外暴露的端口
EXPOSE 19080

#8. 设置启动命令
CMD ["python","service.py"]