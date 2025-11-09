FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl ca-certificates python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Add Ollama to PATH
ENV PATH="/root/.ollama/bin:${PATH}"

# Pull the model after starting Ollama server
RUN ollama serve & \
    sleep 5 && \
    ollama pull gemma3:1b

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Run both FastAPI and Streamlit apps
CMD ["bash", "-c", "ollama serve & sleep 5 && uvicorn main:app --host 0.0.0.0 --port 8000 --reload & streamlit run frontend.py"]
