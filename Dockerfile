FROM python:3.12-slim

# 时区设为中国区，日志时间戳才和本地一致
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先只装依赖，利用 Docker 层缓存：改代码不会触发依赖重装
COPY requirements.txt .
RUN pip install -r requirements.txt

# 再拷代码，顺序很重要
COPY . .

# 用非root 用户跑，降低容器逃逸后的影响面
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 3001

# 健康检查：借/ 接口判断服务是否真的能处理请求
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:3001/').status==200 else 1)"

# 用 exec 形式让 uvicorn 成为 PID 1，能收到 SIGTERM 优雅退出
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3001"]