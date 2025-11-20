import React, { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Play, Download, Github, ExternalLink } from 'lucide-react'
import FileUpload from '../components/FileUpload'
import TextInput from '../components/TextInput'
import AudioUpload from '../components/AudioUpload'
import { modelAPI, metricsAPI } from '../services/api'
import toast from 'react-hot-toast'

const ModelPage = () => {
  const { modelName } = useParams()
  const [modelInfo, setModelInfo] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [predicting, setPredicting] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [textInput, setTextInput] = useState('')
  const [prediction, setPrediction] = useState(null)

  // Load model info and metrics
  useEffect(() => {
    loadModelData()
  }, [modelName])

  const loadModelData = async () => {
    try {
      setLoading(true)
      const [modelResponse, metricsResponse] = await Promise.all([
        modelAPI.getModelInfo(modelName),
        metricsAPI.getMetrics(modelName).catch(() => null),
      ])

      setModelInfo(modelResponse)
      setMetrics(metricsResponse?.metrics)
    } catch (error) {
      toast.error('Failed to load model data: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const handlePredict = async () => {
    if (!selectedFile && !textInput) {
      toast.error('Please provide input')
      return
    }

    try {
      setPredicting(true)
      setPrediction(null)

      const data = {}
      if (selectedFile) data.file = selectedFile
      if (textInput) data.text = textInput

      const result = await modelAPI.predict(modelName, data)
      setPrediction(result.result)
      toast.success('Prediction completed!')
    } catch (error) {
      toast.error('Prediction failed: ' + error.message)
    } finally {
      setPredicting(false)
    }
  }

  const handleClearFile = () => {
    setSelectedFile(null)
    setPrediction(null)
  }

  const handleClearText = () => {
    setTextInput('')
    setPrediction(null)
  }

  const handleTextSubmit = (text) => {
    setTextInput(text)
  }

  // Get accepted file types based on model category
  const getAcceptedTypes = () => {
    if (!modelInfo) return {}

    const category = modelInfo.category
    if (category === 'Computer Vision') {
      return { 'image/*': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'] }
    } else if (category === 'Audio Processing') {
      return { 'audio/*': ['.wav', '.mp3', '.flac', '.ogg'] }
    }
    return {}
  }

  // Determine input type based on model category and name
  const getInputType = () => {
    if (!modelInfo) return 'none'

    const category = modelInfo.category
    const name = modelInfo.name

    // Computer Vision models need image upload
    if (category === 'Computer Vision') return 'image'

    // NLP models need text input
    if (category === 'Natural Language Processing') {
      // Text generation might need different UI than classification
      if (name.includes('generation')) return 'text-generation'
      return 'text'
    }

    // Audio models need audio upload
    if (category === 'Audio Processing') return 'audio'

    // Recommender systems might need user ID + item selection
    if (category === 'Recommender Systems') return 'recommender'

    // Time series might need data upload or manual entry
    if (category === 'Time Series') return 'timeseries'

    // Default
    return 'none'
  }

  const inputType = getInputType()

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (!modelInfo) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Model Not Found</h2>
          <Link to="/" className="text-primary-600 hover:text-primary-700">
            ← Back to Home
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Back Button */}
      <Link
        to="/"
        className="inline-flex items-center space-x-2 text-gray-600 hover:text-gray-900 mb-6"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Back to Models</span>
      </Link>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content - Testing Interface */}
        <div className="lg:col-span-2 space-y-6">
          {/* Model Header */}
          <div className="card">
            <div className="flex items-start justify-between">
              <div>
                <h1 className="text-3xl font-bold text-gray-900 mb-2">
                  {modelInfo.display_name}
                </h1>
                <p className="text-gray-600">{modelInfo.description}</p>
              </div>
              <div className="text-4xl">{modelInfo.icon || '🤖'}</div>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                {modelInfo.category}
              </span>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                modelInfo.status === 'available'
                  ? 'bg-green-100 text-green-800'
                  : 'bg-gray-100 text-gray-800'
              }`}>
                {modelInfo.status}
              </span>
            </div>
          </div>

          {/* Input Section */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Test the Model
            </h2>

            {/* Image Upload for Computer Vision */}
            {inputType === 'image' && (
              <FileUpload
                onFileSelect={setSelectedFile}
                acceptedTypes={getAcceptedTypes()}
                selectedFile={selectedFile}
                onClear={handleClearFile}
              />
            )}

            {/* Text Input for NLP */}
            {(inputType === 'text' || inputType === 'text-generation') && (
              <TextInput
                onTextSubmit={handleTextSubmit}
                placeholder={
                  inputType === 'text-generation'
                    ? "Enter a prompt to generate text..."
                    : "Enter your text here for analysis..."
                }
                maxLength={inputType === 'text-generation' ? 1000 : 5000}
                acceptFiles={true}
              />
            )}

            {/* Audio Upload for Speech Models */}
            {inputType === 'audio' && (
              <AudioUpload
                onFileSelect={setSelectedFile}
                selectedFile={selectedFile}
                onClear={handleClearFile}
                enableRecording={true}
              />
            )}

            {/* Coming Soon Message for Other Types */}
            {['recommender', 'timeseries', 'none'].includes(inputType) && (
              <div className="text-center py-8 bg-gray-50 rounded-lg">
                <p className="text-gray-600 mb-2">
                  Input interface for this model type is coming soon!
                </p>
                <p className="text-sm text-gray-500">
                  Train the model first, then the testing interface will be available.
                </p>
              </div>
            )}

            {/* Predict Button */}
            {!['none', 'recommender', 'timeseries'].includes(inputType) && (
              <button
                onClick={handlePredict}
                disabled={predicting || (!selectedFile && !textInput)}
                className="btn btn-primary w-full mt-4 flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {predicting ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Run Prediction</span>
                  </>
                )}
              </button>
            )}
          </div>

          {/* Results Section */}
          {prediction && (
            <div className="card fade-in">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Prediction Results
              </h2>

              {/* Classification Results (Image, Text, Audio) */}
              {prediction.top5_predictions && (
                <div className="space-y-4">
                  {/* Top Prediction */}
                  <div className="p-4 bg-primary-50 border-2 border-primary-200 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">Top Prediction:</p>
                    <p className="text-2xl font-bold text-primary-900">
                      {prediction.prediction}
                    </p>
                    <p className="text-lg text-primary-700">
                      {(prediction.confidence * 100).toFixed(2)}% confidence
                    </p>
                  </div>

                  {/* Top 5 Predictions */}
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-3">All Predictions:</p>
                    <div className="space-y-2">
                      {prediction.top5_predictions.map((pred, idx) => (
                        <div key={idx} className="flex items-center justify-between">
                          <div className="flex items-center space-x-3 flex-1">
                            <span className="text-2xl">{idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : '•'}</span>
                            <span className="font-medium text-gray-900">{pred.class}</span>
                          </div>
                          <div className="flex items-center space-x-4">
                            <div className="w-48 bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                                style={{ width: `${pred.confidence * 100}%` }}
                              ></div>
                            </div>
                            <span className="text-sm font-medium text-gray-700 w-16 text-right">
                              {(pred.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Text Generation Results */}
              {prediction.generated_text && (
                <div className="space-y-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm font-medium text-gray-700 mb-2">Generated Text:</p>
                    <p className="text-gray-900 whitespace-pre-wrap">{prediction.generated_text}</p>
                  </div>
                  {prediction.tokens && (
                    <p className="text-sm text-gray-500">Tokens: {prediction.tokens}</p>
                  )}
                </div>
              )}

              {/* Transcription Results (ASR) */}
              {prediction.transcription && (
                <div className="space-y-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm font-medium text-gray-700 mb-2">Transcription:</p>
                    <p className="text-lg text-gray-900">{prediction.transcription}</p>
                  </div>
                  {prediction.confidence && (
                    <p className="text-sm text-gray-600">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </p>
                  )}
                </div>
              )}

              {/* Translation Results */}
              {prediction.translation && (
                <div className="space-y-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm font-medium text-gray-700 mb-2">Translation:</p>
                    <p className="text-lg text-gray-900">{prediction.translation}</p>
                  </div>
                  {prediction.source_language && prediction.target_language && (
                    <p className="text-sm text-gray-600">
                      {prediction.source_language} → {prediction.target_language}
                    </p>
                  )}
                </div>
              )}

              {/* Generic JSON Results (fallback) */}
              {!prediction.top5_predictions && !prediction.generated_text && !prediction.transcription && !prediction.translation && (
                <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm">
                  {JSON.stringify(prediction, null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>

        {/* Sidebar - Model Info and Metrics */}
        <div className="space-y-6">
          {/* Model Details */}
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4">Model Details</h3>
            <dl className="space-y-3 text-sm">
              <div>
                <dt className="text-gray-500">Framework</dt>
                <dd className="font-medium text-gray-900 mt-1">
                  {modelInfo.framework || 'PyTorch'}
                </dd>
              </div>
              <div>
                <dt className="text-gray-500">Dataset</dt>
                <dd className="font-medium text-gray-900 mt-1">
                  {modelInfo.dataset || 'N/A'}
                </dd>
              </div>
              {modelInfo.model_size && (
                <div>
                  <dt className="text-gray-500">Model Size</dt>
                  <dd className="font-medium text-gray-900 mt-1">
                    {modelInfo.model_size}
                  </dd>
                </div>
              )}
            </dl>
          </div>

          {/* Metrics */}
          {metrics && (
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-4">Performance Metrics</h3>
              <dl className="space-y-3 text-sm">
                {metrics.final_metrics && Object.entries(metrics.final_metrics).map(([key, value]) => (
                  <div key={key}>
                    <dt className="text-gray-500 capitalize">
                      {key.replace(/_/g, ' ')}
                    </dt>
                    <dd className="font-medium text-gray-900 mt-1">
                      {typeof value === 'number' ? value.toFixed(4) : value}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          )}

          {/* Links */}
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4">Resources</h3>
            <div className="space-y-2">
              <a
                href={`https://github.com/yourusername/ml-portfolio/tree/main/${modelInfo.project_dir}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-sm text-primary-600 hover:text-primary-700"
              >
                <Github className="w-4 h-4" />
                <span>View Source Code</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              <a
                href={`https://github.com/yourusername/ml-portfolio/blob/main/notebooks/${modelInfo.notebook_file}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-sm text-primary-600 hover:text-primary-700"
              >
                <Download className="w-4 h-4" />
                <span>Training Notebook</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ModelPage
