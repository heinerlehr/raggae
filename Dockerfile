# Base image from unstructured
FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
	&& apt-get install -y python3 libmagic-dev poppler-utils tesseract-ocr libreoffice git \
    && apt-get clean # && rm -rf /var/lib/apt/lists/* 
	
ADD https://astral.sh/uv/install.sh /app/uv-installer.sh

# Run the installer then remove it
RUN sh /app/uv-installer.sh && rm /app/uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install Python dependencies
COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml
# Set working directory
WORKDIR /app
# Install Python dependencies
ENV UV_HTTP_TIMEOUT=600
RUN uv sync --frozen

# Expose ports (optional, for debugging or web apps)
EXPOSE 8000

# Set the default command (can be overridden in docker-compose)
CMD ["bash"]
