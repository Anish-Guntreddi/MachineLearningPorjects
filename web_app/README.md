# ML Portfolio - Interactive Model Testing Platform

A comprehensive web application for testing and exploring 12 different machine learning models spanning computer vision, natural language processing, audio processing, recommender systems, time series, and multimodal learning.

## 🌟 Features

- **12 ML Models**: Test models for image classification, object detection, text classification, speech recognition, and more
- **Interactive Testing**: Upload files or enter text directly in the browser to get real-time predictions
- **Metrics Dashboard**: View comprehensive performance metrics, training times, and model comparisons
- **REST API**: FastAPI backend with automatic documentation at `/api/docs`
- **Modern UI**: React frontend with TailwindCSS, responsive design, and smooth animations
- **Docker Support**: One-command deployment with Docker Compose
- **Model Caching**: Efficient model loading to minimize latency
- **Batch Processing**: Process multiple files at once

## 🏗️ Architecture

```
web_app/
├── backend/              # FastAPI backend
│   ├── app.py           # Main application
│   ├── models/          # Model implementations
│   │   ├── base_model.py
│   │   └── image_classifier.py
│   ├── api/             # API endpoints
│   │   ├── inference.py
│   │   └── metrics.py
│   ├── utils/           # Utility functions
│   └── requirements.txt
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # Reusable components
│   │   ├── pages/       # Page components
│   │   ├── services/    # API client
│   │   └── utils/       # Helper functions
│   └── package.json
└── docker-compose.yml   # Docker orchestration
```

## 🚀 Quick Start

**Want to test with your own images right away?** See [QUICK_START.md](QUICK_START.md)

### Prerequisites

- Docker and Docker Compose (recommended)
- OR Python 3.10+ and Node.js 18+ (for local development)

### Option 1: Docker (Recommended)

1. **Navigate to web_app directory**
   ```bash
   cd web_app
   ```

2. **Build and start containers**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - Frontend: http://localhost
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/api/docs

### Option 2: Local Development

#### Backend Setup

```bash
cd web_app/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at http://localhost:8000

#### Frontend Setup

```bash
cd web_app/frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at http://localhost:3000

## 📚 API Documentation

The backend provides automatic API documentation powered by FastAPI:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

### Main Endpoints

#### Models
- `GET /api/models` - List all available models
- `GET /api/models/{model_name}` - Get specific model info

#### Inference
- `POST /api/models/{model_name}/predict` - Make a prediction
- `POST /api/models/{model_name}/batch` - Batch predictions

#### Metrics
- `GET /api/metrics/summary` - Get metrics for all models
- `GET /api/metrics/{model_name}/metrics` - Get model-specific metrics
- `GET /api/metrics/{model_name}/training-history` - Get training history

### Example API Usage

```bash
# Get all models
curl http://localhost:8000/api/models

# Make a prediction (image classification)
curl -X POST \
  -F "file=@test_image.jpg" \
  http://localhost:8000/api/models/image_classification/predict

# Get model metrics
curl http://localhost:8000/api/metrics/image_classification/metrics
```

## 🎯 Available Models

| # | Model | Category | Status |
|---|-------|----------|--------|
| 1 | Image Classification | Computer Vision | ✅ Available |
| 2 | Object Detection | Computer Vision | 🏗️ Coming Soon |
| 3 | Instance Segmentation | Computer Vision | 🏗️ Coming Soon |
| 4 | Text Classification | NLP | 🏗️ Coming Soon |
| 5 | Text Generation | NLP | 🏗️ Coming Soon |
| 6 | Machine Translation | NLP | 🏗️ Coming Soon |
| 7 | Speech Emotion Recognition | Audio | 🏗️ Coming Soon |
| 8 | Automatic Speech Recognition | Audio | 🏗️ Coming Soon |
| 9 | Recommender System | Recommender | 🏗️ Coming Soon |
| 10 | Time Series Forecasting | Time Series | 🏗️ Coming Soon |
| 11 | Anomaly Detection | Anomaly Detection | 🏗️ Coming Soon |
| 12 | Multimodal Fusion | Multimodal | 🏗️ Coming Soon |

## 🛠️ Development

### Backend Development

The backend uses:
- **FastAPI**: Modern async web framework
- **PyTorch**: Deep learning models
- **Pydantic**: Data validation
- **CORS**: Cross-origin support

To add a new model:

1. Create model class in `backend/models/`:
```python
from models.base_model import BaseModel

class NewModel(BaseModel):
    def predict(self, processed_data):
        # Implement prediction logic
        pass
```

2. Register in `backend/models/__init__.py`:
```python
MODEL_REGISTRY = {
    "new_model": NewModel,
    # ...
}
```

### Frontend Development

The frontend uses:
- **React 18**: UI library
- **React Router**: Routing
- **TailwindCSS**: Styling
- **Vite**: Build tool
- **Axios**: HTTP client

To customize:
- Modify components in `src/components/`
- Add pages in `src/pages/`
- Update API client in `src/services/api.js`

### Building for Production

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
npm run build
# Build output in frontend/dist/
```

## 🔧 Configuration

### Environment Variables

Create `.env` file (copy from `.env.example`):

```env
# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
ENVIRONMENT=production

# Frontend
VITE_API_URL=http://localhost:8000/api
```

### Docker Configuration

Modify `docker-compose.yml` for:
- Port mappings
- Volume mounts
- Environment variables
- Resource limits

## 📊 Performance

- **Model Caching**: Models are cached in memory after first load
- **Async Processing**: FastAPI handles multiple requests concurrently
- **Nginx Proxy**: Frontend served via nginx for optimal performance
- **Gzip Compression**: Reduced bandwidth usage
- **Health Checks**: Automatic container health monitoring

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using PyTorch, FastAPI, and React**
