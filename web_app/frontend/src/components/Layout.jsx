import React from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { Brain, Home, BarChart3, Info, Github } from 'lucide-react'

const Layout = () => {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/metrics', label: 'Metrics', icon: BarChart3 },
    { path: '/about', label: 'About', icon: Info },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-sm sticky top-0 z-50">
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2 text-primary-600 hover:text-primary-700 transition-colors">
              <Brain className="w-8 h-8" />
              <span className="text-xl font-bold">ML Portfolio</span>
            </Link>

            {/* Navigation */}
            <div className="flex items-center space-x-6">
              {navItems.map(({ path, label, icon: Icon }) => {
                const isActive = location.pathname === path
                return (
                  <Link
                    key={path}
                    to={path}
                    className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-primary-50 text-primary-700'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{label}</span>
                  </Link>
                )
              })}

              {/* GitHub Link */}
              <a
                href="https://github.com/yourusername/ml-portfolio"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-100 transition-colors"
              >
                <Github className="w-4 h-4" />
                <span>GitHub</span>
              </a>
            </div>
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-300 py-8 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* About */}
            <div>
              <h3 className="text-white font-semibold mb-3">ML Portfolio</h3>
              <p className="text-sm text-gray-400">
                Interactive platform for testing 12 different machine learning models
                spanning computer vision, NLP, audio processing, and more.
              </p>
            </div>

            {/* Quick Links */}
            <div>
              <h3 className="text-white font-semibold mb-3">Quick Links</h3>
              <ul className="space-y-2 text-sm">
                <li>
                  <Link to="/" className="hover:text-white transition-colors">
                    All Models
                  </Link>
                </li>
                <li>
                  <Link to="/metrics" className="hover:text-white transition-colors">
                    Metrics Dashboard
                  </Link>
                </li>
                <li>
                  <Link to="/about" className="hover:text-white transition-colors">
                    About
                  </Link>
                </li>
              </ul>
            </div>

            {/* Technologies */}
            <div>
              <h3 className="text-white font-semibold mb-3">Technologies</h3>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>PyTorch • TensorFlow</li>
                <li>Hugging Face Transformers</li>
                <li>FastAPI • React</li>
                <li>Docker • Kubernetes</li>
              </ul>
            </div>
          </div>

          <div className="mt-8 pt-8 border-t border-gray-700 text-center text-sm text-gray-400">
            <p>&copy; {new Date().getFullYear()} ML Portfolio. Built with ❤️ for learning.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default Layout
