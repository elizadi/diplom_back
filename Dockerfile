# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.12.0-gpu
COPY . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --no-cache-dir --upgrade pip -r requirements.txt 
# RUN chmod +x requirements.txt
# RUN chmod +x manage.py
EXPOSE 8000 
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# wsl --shutdown
# wsl --export docker-desktop-data D:\docker\docker-desktop-data.tar
# wsl --unregister docker-desktop-data
# wsl --import docker-desktop-data  D:\docker D:\docker\docker-desktop-data.tar --version 2
# https://iximiuz.com/en/posts/docker-publish-port-of-running-container/