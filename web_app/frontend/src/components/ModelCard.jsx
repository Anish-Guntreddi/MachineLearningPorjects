import React from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, CheckCircle, Clock, AlertCircle } from 'lucide-react'

const ModelCard = ({ model }) => {
  const {
    name,
    display_name,
    description,
    category,
    status,
    metrics_available,
    icon,
  } = model

  // Status configurations
  const statusConfig = {
    available: {
      icon: CheckCircle,
      text: 'Available',
      color: 'text-green-600',
      bg: 'bg-green-50',
    },
    training: {
      icon: Clock,
      text: 'Training',
      color: 'text-yellow-600',
      bg: 'bg-yellow-50',
    },
    unavailable: {
      icon: AlertCircle,
      text: 'Unavailable',
      color: 'text-gray-600',
      bg: 'bg-gray-50',
    },
  }

  const statusInfo = statusConfig[status] || statusConfig.unavailable
  const StatusIcon = statusInfo.icon

  // Category colors
  const categoryColors = {
    'Computer Vision': 'bg-blue-100 text-blue-800',
    'Natural Language Processing': 'bg-purple-100 text-purple-800',
    'Audio Processing': 'bg-pink-100 text-pink-800',
    'Recommender Systems': 'bg-orange-100 text-orange-800',
    'Time Series': 'bg-teal-100 text-teal-800',
    'Anomaly Detection': 'bg-red-100 text-red-800',
    'Multimodal': 'bg-indigo-100 text-indigo-800',
  }

  return (
    <Link
      to={`/model/${name}`}
      className="card group hover:scale-[1.02] transition-all duration-200"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          {/* Icon */}
          <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center text-primary-600 text-2xl">
            {icon || '🤖'}
          </div>

          {/* Title */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">
              {display_name}
            </h3>
            <div className="flex items-center space-x-2 mt-1">
              <StatusIcon className={`w-4 h-4 ${statusInfo.color}`} />
              <span className={`text-xs font-medium ${statusInfo.color}`}>
                {statusInfo.text}
              </span>
            </div>
          </div>
        </div>

        {/* Arrow */}
        <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-primary-600 group-hover:translate-x-1 transition-all" />
      </div>

      {/* Category Badge */}
      <div className="mb-3">
        <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${categoryColors[category] || 'bg-gray-100 text-gray-800'}`}>
          {category}
        </span>
      </div>

      {/* Description */}
      <p className="text-sm text-gray-600 line-clamp-2 mb-4">
        {description}
      </p>

      {/* Footer */}
      <div className="flex items-center justify-between pt-4 border-t border-gray-100">
        <div className="flex items-center space-x-4 text-xs text-gray-500">
          {metrics_available ? (
            <span className="flex items-center space-x-1">
              <CheckCircle className="w-3 h-3 text-green-500" />
              <span>Metrics Available</span>
            </span>
          ) : (
            <span className="flex items-center space-x-1">
              <AlertCircle className="w-3 h-3 text-gray-400" />
              <span>No Metrics</span>
            </span>
          )}
        </div>

        <span className="text-xs font-medium text-primary-600 group-hover:underline">
          Test Model →
        </span>
      </div>
    </Link>
  )
}

export default ModelCard
