FROM python:3.9-slim
# Set the working directory
WORKDIR /linearregression-app
# Copy the requirements file
COPY requirements.txt requirements.txt
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the application files
COPY app.py app.py
COPY linear_model.pkl linear_model.pkl
# Expose the port the app runs on
EXPOSE 5000
# Define the command to run the application
CMD ["python", "app.py"]
#docker build -t linearregression-app .
#docker run -p 5000:5000 --name linearregressionflaskapp linearregression-app
#test using postman
#POST http://localhost:5000/predict
#Body -> raw -> JSON
#{
#  "features": [8.3252, 41, 6.984127, 1.02381, 322, 2.555556, 37.88, -122.23]
#}

