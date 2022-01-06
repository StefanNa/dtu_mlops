FROM anibali/pytorch:1.8.1-cuda11.1


WORKDIR / app/ -> WORKDIR /app

COPY pytorch_docker.py pytorch_docker.py

ENTRYPOINT ["python", "-u", "pytorch_docker.py"]

