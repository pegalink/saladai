FROM ollama/ollama:latest

# Set host so Ollama listens on all interfaces (fixes the 503 error)
ENV OLLAMA_HOST=0.0.0.0:11434

# Start Ollama in the background, wait for it to boot, pull the model, then shut down
RUN ollama serve & \
    sleep 5 && \
    ollama pull ornith:9b

# Expose port
EXPOSE 11434

# Start Ollama when container runs
ENTRYPOINT ["ollama", "serve"]
