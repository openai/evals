FROM python:3.9-slim

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1

# Install pipenv and compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev libssl-dev parallel --quiet

WORKDIR /app

# Install application into container
COPY . .
RUN pip install --upgrade pip setuptools wheel --no-cache-dir

RUN pip install -e . --no-cache-dir

# Run the application from docker-compose.yml in command argument
# CMD ["python", "main.py"]
