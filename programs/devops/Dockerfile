FROM python:3.9-slim
# Set the working directory
WORKDIR /flask-app
# Copy the requirements file
COPY requirements.txt requirements.txt
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the application files
COPY app.py app.py
# Expose the port the app runs on
EXPOSE 5000
# Define the command to run the application
CMD ["python", "app.py"]
#docker build -t flaskapp .
#docker run -p 5005:5000 flaskapp  
#docker run -d -p 8000:5000 --name myflaskapp1 flaskapp
#docker exec -it myflaskapp1 /bin/bash

