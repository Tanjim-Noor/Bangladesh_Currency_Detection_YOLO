# Deployment Guide

## Overview

This guide covers deploying the Bangladeshi Taka Detection API in various environments.

---

## Prerequisites

- Docker 20.10+ (for containerized deployment)
- Python 3.11+ (for local deployment)
- At least 2GB RAM available
- Model weights file (`best.pt`)

---

## Deployment Options

### 1. Local Development

```bash
# Clone/navigate to project
cd Bangladesh_Currency_Detection_YOLO

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start API
cd phase2
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Docker Deployment

```bash
# From project root
cd Bangladesh_Currency_Detection_YOLO

# Build image
docker build -t bd-taka-detector -f phase2/docker/Dockerfile .

# Run container
docker run -d \
  --name taka-api \
  -p 8000:8000 \
  -e CONFIDENCE_THRESHOLD=0.25 \
  bd-taka-detector

# Check logs
docker logs -f taka-api

# Stop container
docker stop taka-api
docker rm taka-api
```

### 3. Docker Compose Deployment

```bash
cd phase2/docker

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Environment Configuration

Copy `deployment/ENV_TEMPLATE` to `.env` and configure:

```bash
cp deployment/ENV_TEMPLATE .env
```

Key variables:
- `MODEL_PATH`: Path to YOLO weights
- `CONFIDENCE_THRESHOLD`: Detection threshold (0.0-1.0)
- `API_PORT`: Server port (default: 8000)

---

## Health Monitoring

The API exposes a health check endpoint:

```bash
curl http://localhost:8000/health
```

For Docker, the container includes a built-in health check that runs every 30 seconds.

---

## Scaling Considerations

### Single Instance
- Suitable for development and low traffic
- Default configuration

### Multiple Workers (Uvicorn)
```bash
uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Load Balancer
For high-traffic production:
1. Deploy multiple container instances
2. Use nginx or traefik as load balancer
3. Consider using Kubernetes for orchestration

---

## Security Recommendations

1. **HTTPS**: Use reverse proxy with SSL/TLS
2. **Rate Limiting**: Implement at proxy level
3. **Authentication**: Add API key validation if needed
4. **CORS**: Configure allowed origins in production
5. **File Size**: Set appropriate upload limits

---

## Troubleshooting

### Model Not Loading
- Verify `best.pt` exists at configured path
- Check file permissions
- Ensure sufficient memory available

### Container Won't Start
- Check Docker logs: `docker logs <container>`
- Verify port 8000 is available
- Ensure image built successfully

### Slow Inference
- First request is slow (model warm-up)
- Consider CPU vs GPU deployment
- Check available system resources

---

## Support

For issues, check:
1. [Phase 2 README](../README_PHASE2.md)
2. [API Documentation](API_DOCUMENTATION.md)
3. Project issues on GitHub
