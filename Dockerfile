FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port the server will run on
EXPOSE 8080

# Start the FastAPI server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]