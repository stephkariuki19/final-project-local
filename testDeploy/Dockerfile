FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Set working directory inside the container
WORKDIR /app

# Copy the Python script and any necessary files
COPY ./app /app

# Install any dependencies
RUN pip install fastapi uvicorn

# Expose the port where the FastAPI server will run
EXPOSE 80

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
