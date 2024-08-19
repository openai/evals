FROM python:3.8-slim-buster

# install wget for container ready check
RUN apt-get update && apt-get install -y wget
WORKDIR /app
COPY . .
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN pip3 install -r requirements.txt

ENTRYPOINT ["/docker-entrypoint.sh"]