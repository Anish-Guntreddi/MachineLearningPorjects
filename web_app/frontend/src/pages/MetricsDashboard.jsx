import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { BarChart3, TrendingUp, Award, Clock } from 'lucide-react'
import { metricsAPI } from '../services/api'
import toast from 'react-hot-toast'

const MetricsDashboard = () => {
  const [metricsSummary, setMetricsSummary] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadMetricsSummary()
  }, [])

  const loadMetricsSummary = async () => {
    try {
      setLoading(true)
      const data = await metricsAPI.getSummary()
      setMetricsSummary(data)
    } catch (error) {
      toast.error('Failed to load metrics: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  const models = metricsSummary?.models || []
  const availableModels = models.filter((m) => m.status === 'available')

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Metrics Dashboard
        </h1>
        <p className="text-gray-600">
          Performance metrics and evaluation results for all trained models
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-2 bg-blue-100 rounded-lg">
              <BarChart3 className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Models</p>
              <p className="text-2xl font-bold text-gray-900">
                {metricsSummary?.total_models || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-2 bg-green-100 rounded-lg">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">With Metrics</p>
              <p className="text-2xl font-bold text-gray-900">
                {availableModels.length}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Award className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Avg Accuracy</p>
              <p className="text-2xl font-bold text-gray-900">
                {availableModels.length > 0
                  ? (
                      availableModels
                        .filter((m) => m.accuracy)
                        .reduce((sum, m) => sum + (m.accuracy || 0), 0) /
                      availableModels.filter((m) => m.accuracy).length
                    ).toFixed(2) + '%'
                  : 'N/A'}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-2 bg-orange-100 rounded-lg">
              <Clock className="w-6 h-6 text-orange-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Training Time</p>
              <p className="text-2xl font-bold text-gray-900">
                {availableModels.length > 0
                  ? (
                      availableModels.reduce((sum, m) => sum + (m.training_time_minutes || 0), 0)
                    ).toFixed(0) + ' min'
                  : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Models Table */}
      <div className="card overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Model Performance</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Dataset
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  F1 Score
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Parameters
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Training Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {models.map((model) => (
                <tr key={model.model_name} className="hover:bg-gray-50 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Link
                      to={`/model/${model.model_name}`}
                      className="text-primary-600 hover:text-primary-700 font-medium"
                    >
                      {model.model_name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                    </Link>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {model.dataset || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {model.accuracy ? (
                      <div className="flex items-center space-x-2">
                        <div className="w-20 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${model.accuracy}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-gray-700">
                          {model.accuracy.toFixed(2)}%
                        </span>
                      </div>
                    ) : (
                      <span className="text-sm text-gray-400">N/A</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {model.f1_score ? model.f1_score.toFixed(4) : 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {model.parameters
                      ? (model.parameters / 1000000).toFixed(2) + 'M'
                      : 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {model.training_time_minutes
                      ? model.training_time_minutes.toFixed(1) + ' min'
                      : 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        model.status === 'available'
                          ? 'bg-green-100 text-green-800'
                          : model.status === 'error'
                          ? 'bg-red-100 text-red-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {model.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {models.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">No metrics available</p>
          </div>
        )}
      </div>

      {/* Additional Metrics Cards */}
      {availableModels.length > 0 && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Best Performing Model */}
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4 flex items-center space-x-2">
              <Award className="w-5 h-5 text-yellow-500" />
              <span>Best Performing Model</span>
            </h3>
            {(() => {
              const bestModel = availableModels.reduce((best, current) => {
                return (current.accuracy || 0) > (best.accuracy || 0) ? current : best
              }, availableModels[0])

              return (
                <div>
                  <Link
                    to={`/model/${bestModel.model_name}`}
                    className="text-lg font-medium text-primary-600 hover:text-primary-700"
                  >
                    {bestModel.model_name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                  </Link>
                  <p className="text-3xl font-bold text-gray-900 mt-2">
                    {bestModel.accuracy?.toFixed(2)}% accuracy
                  </p>
                  <p className="text-sm text-gray-500 mt-1">{bestModel.dataset}</p>
                </div>
              )
            })()}
          </div>

          {/* Fastest Training */}
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4 flex items-center space-x-2">
              <Clock className="w-5 h-5 text-blue-500" />
              <span>Fastest Training</span>
            </h3>
            {(() => {
              const fastestModel = availableModels
                .filter((m) => m.training_time_minutes)
                .reduce((fastest, current) => {
                  return (current.training_time_minutes || Infinity) < (fastest.training_time_minutes || Infinity)
                    ? current
                    : fastest
                }, availableModels[0])

              return (
                <div>
                  <Link
                    to={`/model/${fastestModel.model_name}`}
                    className="text-lg font-medium text-primary-600 hover:text-primary-700"
                  >
                    {fastestModel.model_name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                  </Link>
                  <p className="text-3xl font-bold text-gray-900 mt-2">
                    {fastestModel.training_time_minutes?.toFixed(1)} minutes
                  </p>
                  <p className="text-sm text-gray-500 mt-1">{fastestModel.dataset}</p>
                </div>
              )
            })()}
          </div>
        </div>
      )}
    </div>
  )
}

export default MetricsDashboard
