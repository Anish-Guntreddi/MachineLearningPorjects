import React, { useState, useEffect } from 'react'
import { Search, Filter } from 'lucide-react'
import ModelCard from '../components/ModelCard'
import { modelAPI } from '../services/api'
import toast from 'react-hot-toast'

const HomePage = () => {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')

  // Load models on mount
  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      setLoading(true)
      const response = await modelAPI.getAllModels()
      setModels(response.models || [])
    } catch (error) {
      toast.error('Failed to load models: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  // Get unique categories
  const categories = ['all', ...new Set(models.map((m) => m.category))]

  // Filter models
  const filteredModels = models.filter((model) => {
    const matchesSearch =
      model.display_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      model.description.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesCategory =
      selectedCategory === 'all' || model.category === selectedCategory

    return matchesSearch && matchesCategory
  })

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Interactive ML Model Testing Platform
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Explore and test 12 different machine learning models spanning computer vision,
          natural language processing, audio processing, recommender systems, and more.
        </p>
      </div>

      {/* Search and Filter */}
      <div className="mb-8">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search models..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input pl-10"
            />
          </div>

          {/* Category Filter */}
          <div className="sm:w-64 relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="input pl-10 appearance-none cursor-pointer"
            >
              {categories.map((category) => (
                <option key={category} value={category}>
                  {category === 'all' ? 'All Categories' : category}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Results Count */}
      <div className="mb-6 text-sm text-gray-600">
        Showing {filteredModels.length} of {models.length} models
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex justify-center items-center py-20">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      )}

      {/* Models Grid */}
      {!loading && filteredModels.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 fade-in">
          {filteredModels.map((model) => (
            <ModelCard key={model.name} model={model} />
          ))}
        </div>
      )}

      {/* No Results */}
      {!loading && filteredModels.length === 0 && (
        <div className="text-center py-20">
          <p className="text-gray-600 text-lg mb-2">No models found</p>
          <p className="text-gray-500">
            Try adjusting your search or filter criteria
          </p>
        </div>
      )}

      {/* Stats Section */}
      {!loading && models.length > 0 && (
        <div className="mt-16 grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="card text-center">
            <div className="text-3xl font-bold text-primary-600 mb-2">
              {models.length}
            </div>
            <div className="text-gray-600">Total Models</div>
          </div>

          <div className="card text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">
              {models.filter((m) => m.status === 'available').length}
            </div>
            <div className="text-gray-600">Available</div>
          </div>

          <div className="card text-center">
            <div className="text-3xl font-bold text-purple-600 mb-2">
              {categories.length - 1}
            </div>
            <div className="text-gray-600">Categories</div>
          </div>

          <div className="card text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">
              {models.filter((m) => m.metrics_available).length}
            </div>
            <div className="text-gray-600">With Metrics</div>
          </div>
        </div>
      )}
    </div>
  )
}

export default HomePage
