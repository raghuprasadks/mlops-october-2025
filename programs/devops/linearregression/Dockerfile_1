FROM python:3.9-slim
# Set the working directory
WORKDIR /linearregression-app
# Copy the requirements file
COPY requirements.txt requirements.txt
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the application files
COPY linearregression.py linearregression.py
# Expose the port the app runs on
#EXPOSE 5000
# Define the command to run the application
CMD ["python", "linearregression.py"]
#docker build -t linearregression-app .
#docker run linearregression-app  

