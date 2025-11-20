/**
 * API Service
 * Handles all backend API communication
 */

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens here if needed
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    // Handle errors globally
    const message = error.response?.data?.detail || error.message || 'An error occurred'
    return Promise.reject(new Error(message))
  }
)

/**
 * Model API
 */
export const modelAPI = {
  // Get all models
  getAllModels: () => api.get('/models'),

  // Get specific model info
  getModelInfo: (modelName) => api.get(`/models/${modelName}`),

  // Make prediction
  predict: (modelName, data) => {
    const formData = new FormData()

    // Handle different data types
    if (data.file) {
      formData.append('file', data.file)
    }
    if (data.text) {
      formData.append('text', data.text)
    }
    if (data.audio) {
      formData.append('audio', data.audio)
    }

    return api.post(`/models/${modelName}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },

  // Batch prediction
  predictBatch: (modelName, files) => {
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })

    return api.post(`/models/${modelName}/batch`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },
}

/**
 * Metrics API
 */
export const metricsAPI = {
  // Get model metrics
  getMetrics: (modelName) => api.get(`/metrics/${modelName}/metrics`),

  // Get all metrics summary
  getSummary: () => api.get('/metrics/summary'),

  // Get training history
  getTrainingHistory: (modelName) => api.get(`/metrics/${modelName}/training-history`),

  // Get visualizations
  getVisualizations: (modelName) => api.get(`/metrics/${modelName}/visualizations`),
}

/**
 * Health check
 */
export const healthAPI = {
  check: () => api.get('/health'),
  getDeviceInfo: () => api.get('/device-info'),
}

export default api
