#!/bin/bash
# Startup script for ML Portfolio Web Application

set -e

echo "=========================================="
echo "ML Portfolio - Interactive Testing Platform"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose found"
echo ""

# Check if .env exists, if not create from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file. Please review and update if needed."
    echo ""
fi

# Build and start containers
echo "🏗️  Building Docker containers..."
echo "This may take several minutes on first run..."
echo ""

docker-compose build

echo ""
echo "🚀 Starting containers..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "=========================================="
    echo "✅ Application is running!"
    echo "=========================================="
    echo ""
    echo "📱 Frontend: http://localhost"
    echo "🔧 Backend API: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/api/docs"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f"
    echo ""
    echo "To stop the application:"
    echo "  docker-compose down"
    echo ""
    echo "=========================================="
else
    echo ""
    echo "❌ Failed to start services. Check logs with:"
    echo "  docker-compose logs"
    exit 1
fi
