# Use an official lightweight Python image.
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the packages
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "collect-github_data.py"]
