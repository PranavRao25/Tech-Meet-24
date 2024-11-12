# Use an official Python runtime as a base image
FROM python:3.10

# Set environment variables to avoid issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    libprotobuf-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Allow deprecated sklearn install if you are still using it in some dependencies
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Expose Streamlit’s default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "User_Interface/interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
