# Deployment Guide

This guide covers deploying the ML Portfolio web application to various environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Best Practices](#production-best-practices)
5. [Troubleshooting](#troubleshooting)

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### Backend Setup

```bash
cd web_app/backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at http://localhost:8000

### Frontend Setup

```bash
cd web_app/frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend will be available at http://localhost:3000

## Docker Deployment

### Quick Start

```bash
cd web_app

# Option 1: Use the startup script
./start.sh

# Option 2: Manual Docker Compose
docker-compose up --build
```

Access the application:
- Frontend: http://localhost
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

### Docker Commands

```bash
# Build containers
docker-compose build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down

# Restart services
docker-compose restart

# Remove all data
docker-compose down -v
```

### Production Docker Configuration

For production, modify `docker-compose.yml`:

```yaml
services:
  backend:
    environment:
      - ENVIRONMENT=production
    # Remove volume mounts for code hot-reload
    # Add resource limits
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Cloud Deployment

### AWS Deployment

#### Using EC2

1. **Launch EC2 instance**
   - Choose Ubuntu 22.04 LTS
   - Minimum: t3.large (2 vCPUs, 8GB RAM)
   - Recommended: t3.xlarge or GPU instance for faster inference
   - Open ports: 80 (HTTP), 443 (HTTPS), 22 (SSH)

2. **SSH into instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install Docker**
   ```bash
   sudo apt update
   sudo apt install -y docker.io docker-compose
   sudo usermod -aG docker $USER
   ```

4. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/ml-portfolio.git
   cd ml-portfolio/web_app
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with production values
   nano .env
   ```

6. **Start application**
   ```bash
   ./start.sh
   ```

#### Using ECS (Elastic Container Service)

1. **Push images to ECR**
   ```bash
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_URL

   # Tag and push backend
   docker tag ml-portfolio-backend:latest YOUR_ECR_URL/ml-portfolio-backend:latest
   docker push YOUR_ECR_URL/ml-portfolio-backend:latest

   # Tag and push frontend
   docker tag ml-portfolio-frontend:latest YOUR_ECR_URL/ml-portfolio-frontend:latest
   docker push YOUR_ECR_URL/ml-portfolio-frontend:latest
   ```

2. **Create ECS task definitions** (use AWS Console or CLI)

3. **Deploy to ECS cluster**

### Google Cloud Platform

#### Using Compute Engine

Similar to AWS EC2 - follow the EC2 steps with GCP-specific commands.

#### Using Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-portfolio-backend backend/
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-portfolio-frontend frontend/

# Deploy backend
gcloud run deploy ml-portfolio-backend \
  --image gcr.io/PROJECT_ID/ml-portfolio-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2

# Deploy frontend
gcloud run deploy ml-portfolio-frontend \
  --image gcr.io/PROJECT_ID/ml-portfolio-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Deployment

#### Using Azure Container Instances

```bash
# Login to Azure
az login

# Create resource group
az group create --name ml-portfolio-rg --location eastus

# Deploy containers
az container create \
  --resource-group ml-portfolio-rg \
  --name ml-portfolio \
  --image YOUR_ACR/ml-portfolio-backend:latest \
  --dns-name-label ml-portfolio \
  --ports 8000
```

### DigitalOcean App Platform

1. Connect GitHub repository
2. Configure build and run commands
3. Set environment variables
4. Deploy

## Production Best Practices

### Security

1. **Use HTTPS**
   - Obtain SSL certificate (Let's Encrypt)
   - Configure nginx for HTTPS

   ```nginx
   server {
       listen 443 ssl http2;
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       # ... rest of config
   }
   ```

2. **Environment Variables**
   - Never commit `.env` file
   - Use secrets management (AWS Secrets Manager, etc.)
   - Rotate API keys regularly

3. **Rate Limiting**
   - Add rate limiting to prevent abuse
   - Use nginx limit_req or application-level limiting

4. **CORS Configuration**
   - Restrict CORS to specific origins
   - Don't use wildcard (*) in production

### Performance

1. **Model Optimization**
   - Use quantization for faster inference
   - Consider ONNX runtime for production
   - Implement model versioning

2. **Caching**
   - Enable Redis for model caching
   - Cache API responses where appropriate
   - Use CDN for static assets

3. **Load Balancing**
   - Use nginx or cloud load balancer
   - Scale horizontally with multiple backend instances
   - Implement health checks

4. **Monitoring**
   - Set up logging (ELK stack, CloudWatch, etc.)
   - Monitor metrics (Prometheus + Grafana)
   - Set up alerts for errors and performance issues

### Database (if needed)

If you add user authentication or result storage:

```yaml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mlportfolio
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill -9 PID

# Or change port in docker-compose.yml
ports:
  - "8001:8000"
```

#### Out of Memory

```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory

# Or reduce model batch size in code
```

#### Models Not Loading

```bash
# Check volume mounts in docker-compose.yml
# Ensure model files exist in project directories
ls -la ../01_Image_Classification/models/

# Check container logs
docker-compose logs backend
```

#### CORS Errors

Update backend CORS configuration in `app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### Container Health Check Failing

```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs backend

# Restart unhealthy container
docker-compose restart backend
```

### Debugging

```bash
# Access running container
docker-compose exec backend bash

# Check Python environment
docker-compose exec backend python --version
docker-compose exec backend pip list

# Test API endpoint manually
curl http://localhost:8000/api/health
```

## Scaling

### Horizontal Scaling

```yaml
services:
  backend:
    deploy:
      replicas: 3

  nginx:
    image: nginx:alpine
    depends_on:
      - backend
    ports:
      - "80:80"
```

### Vertical Scaling

Increase resources per container:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

## Backup and Recovery

```bash
# Backup models and results
tar -czf ml_portfolio_backup.tar.gz \
  ../01_Image_Classification/models \
  ../01_Image_Classification/results \
  # ... other projects

# Restore from backup
tar -xzf ml_portfolio_backup.tar.gz
```

## Updates and Maintenance

```bash
# Pull latest code
git pull origin main

# Rebuild containers
docker-compose build

# Restart with zero downtime
docker-compose up -d --no-deps --build backend
docker-compose up -d --no-deps --build frontend
```

## Monitoring and Logging

### Application Logs

```bash
# View real-time logs
docker-compose logs -f

# View specific service
docker-compose logs -f backend

# Export logs
docker-compose logs > application.log
```

### Resource Monitoring

```bash
# Container stats
docker stats

# Disk usage
docker system df
```

---

For additional support, please refer to the main [README.md](README.md) or open an issue on GitHub.
