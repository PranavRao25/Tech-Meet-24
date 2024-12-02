# Use an appropriate base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy all necessary files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the run.sh script executable
RUN chmod +x run.sh

# Expose port for Streamlit
EXPOSE 8501

# Run the database setup and Streamlit app
ENTRYPOINT ["./run.sh"]
