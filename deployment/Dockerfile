# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container to /deployment
WORKDIR /deployment

# Copy the current directory contents into the container at /deployment
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8080

# Define environment variable for where the application should look for the 'src' directory
ENV PYTHONPATH "${PYTHONPATH}:/deployment/src"

# Run main.py when the container launches
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
