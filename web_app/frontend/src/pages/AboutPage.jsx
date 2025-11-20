import React from 'react'
import { Github, Linkedin, Mail, Code, Brain, Rocket } from 'lucide-react'

const AboutPage = () => {
  const technologies = {
    'Deep Learning': ['PyTorch', 'TensorFlow', 'Hugging Face Transformers', 'timm'],
    'Computer Vision': ['OpenCV', 'torchvision', 'albumentations', 'Detectron2'],
    'NLP': ['BERT', 'GPT', 'T5', 'spaCy', 'nltk'],
    'Audio': ['librosa', 'torchaudio', 'Wav2Vec2', 'Whisper'],
    'Web Development': ['FastAPI', 'React', 'Vite', 'TailwindCSS'],
    'Deployment': ['Docker', 'Uvicorn', 'Nginx'],
  }

  const projects = [
    {
      name: 'Image Classification',
      description: 'CIFAR-10 classification with CNNs and ResNets',
      category: 'Computer Vision',
    },
    {
      name: 'Object Detection',
      description: 'COCO object detection with YOLO/Faster R-CNN',
      category: 'Computer Vision',
    },
    {
      name: 'Instance Segmentation',
      description: 'Mask R-CNN for instance segmentation',
      category: 'Computer Vision',
    },
    {
      name: 'Text Classification',
      description: 'Sentiment analysis with BERT',
      category: 'NLP',
    },
    {
      name: 'Text Generation',
      description: 'GPT-based text generation',
      category: 'NLP',
    },
    {
      name: 'Machine Translation',
      description: 'Seq2seq translation models',
      category: 'NLP',
    },
    {
      name: 'Speech Emotion Recognition',
      description: 'Emotion recognition from audio',
      category: 'Audio',
    },
    {
      name: 'Automatic Speech Recognition',
      description: 'Speech-to-text with Wav2Vec2',
      category: 'Audio',
    },
    {
      name: 'Recommender System',
      description: 'MovieLens recommendations',
      category: 'Recommender',
    },
    {
      name: 'Time Series Forecasting',
      description: 'LSTM/Transformer for time series',
      category: 'Time Series',
    },
    {
      name: 'Anomaly Detection',
      description: 'Autoencoder-based anomaly detection',
      category: 'Anomaly Detection',
    },
    {
      name: 'Multimodal Fusion',
      description: 'Multi-modal learning',
      category: 'Multimodal',
    },
  ]

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          About This Project
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          A comprehensive machine learning portfolio showcasing 12 end-to-end projects
          with interactive testing capabilities.
        </p>
      </div>

      {/* Overview */}
      <div className="card mb-12">
        <div className="flex items-start space-x-4 mb-6">
          <div className="p-3 bg-primary-100 rounded-lg">
            <Brain className="w-8 h-8 text-primary-600" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Project Overview</h2>
            <p className="text-gray-600">
              This platform demonstrates proficiency across multiple domains of machine learning,
              from computer vision to natural language processing, audio processing, and beyond.
              Each project includes complete implementations with training code, evaluation metrics,
              and interactive testing interfaces.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
          <div className="text-center p-6 bg-blue-50 rounded-lg">
            <Code className="w-12 h-12 text-blue-600 mx-auto mb-3" />
            <h3 className="font-semibold text-gray-900 mb-2">12 Projects</h3>
            <p className="text-sm text-gray-600">
              Covering major ML domains and use cases
            </p>
          </div>
          <div className="text-center p-6 bg-purple-50 rounded-lg">
            <Brain className="w-12 h-12 text-purple-600 mx-auto mb-3" />
            <h3 className="font-semibold text-gray-900 mb-2">Production-Ready</h3>
            <p className="text-sm text-gray-600">
              Dockerized deployment with FastAPI backend
            </p>
          </div>
          <div className="text-center p-6 bg-green-50 rounded-lg">
            <Rocket className="w-12 h-12 text-green-600 mx-auto mb-3" />
            <h3 className="font-semibold text-gray-900 mb-2">Interactive</h3>
            <p className="text-sm text-gray-600">
              Test models directly in your browser
            </p>
          </div>
        </div>
      </div>

      {/* Technologies */}
      <div className="card mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Technologies Used</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Object.entries(technologies).map(([category, tools]) => (
            <div key={category}>
              <h3 className="font-semibold text-gray-900 mb-3">{category}</h3>
              <ul className="space-y-2">
                {tools.map((tool) => (
                  <li key={tool} className="flex items-center space-x-2 text-sm text-gray-600">
                    <span className="w-1.5 h-1.5 bg-primary-600 rounded-full"></span>
                    <span>{tool}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* All Projects */}
      <div className="card mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">All Projects</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {projects.map((project, idx) => (
            <div key={idx} className="border border-gray-200 rounded-lg p-4 hover:border-primary-300 transition-colors">
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-primary-100 text-primary-600 rounded-lg flex items-center justify-center font-semibold text-sm flex-shrink-0">
                  {idx + 1}
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{project.name}</h3>
                  <p className="text-sm text-gray-600 mt-1">{project.description}</p>
                  <span className="inline-block mt-2 px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">
                    {project.category}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Features */}
      <div className="card mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">📓 Jupyter Notebooks</h3>
            <p className="text-gray-600 text-sm">
              Complete training notebooks with CUDA/CPU support, data loading,
              model training, evaluation, and visualization.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">🚀 REST API</h3>
            <p className="text-gray-600 text-sm">
              FastAPI backend with automatic documentation, file upload support,
              and batch processing capabilities.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">💻 Interactive UI</h3>
            <p className="text-gray-600 text-sm">
              React frontend with real-time predictions, metrics visualization,
              and responsive design.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">📊 Metrics Dashboard</h3>
            <p className="text-gray-600 text-sm">
              Comprehensive dashboard showing model performance, training times,
              and evaluation metrics.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">🐳 Docker Support</h3>
            <p className="text-gray-600 text-sm">
              Containerized deployment with Docker Compose for easy setup
              and scalability.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">📈 Model Caching</h3>
            <p className="text-gray-600 text-sm">
              Efficient model loading with caching to reduce inference latency
              and improve performance.
            </p>
          </div>
        </div>
      </div>

      {/* Contact/Links */}
      <div className="card text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Connect</h2>
        <div className="flex justify-center space-x-6">
          <a
            href="https://github.com/yourusername"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-2 text-gray-700 hover:text-primary-600 transition-colors"
          >
            <Github className="w-6 h-6" />
            <span>GitHub</span>
          </a>
          <a
            href="https://linkedin.com/in/yourusername"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-2 text-gray-700 hover:text-primary-600 transition-colors"
          >
            <Linkedin className="w-6 h-6" />
            <span>LinkedIn</span>
          </a>
          <a
            href="mailto:your.email@example.com"
            className="flex items-center space-x-2 text-gray-700 hover:text-primary-600 transition-colors"
          >
            <Mail className="w-6 h-6" />
            <span>Email</span>
          </a>
        </div>
      </div>
    </div>
  )
}

export default AboutPage
