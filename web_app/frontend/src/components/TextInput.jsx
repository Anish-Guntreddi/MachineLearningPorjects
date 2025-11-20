import React, { useState } from 'react'
import { Type, X, FileText } from 'lucide-react'
import clsx from 'clsx'

const TextInput = ({
  onTextSubmit,
  placeholder = "Enter your text here...",
  maxLength = 5000,
  minRows = 4,
  maxRows = 12,
  showCharCount = true,
  acceptFiles = false,
}) => {
  const [text, setText] = useState('')
  const [rows, setRows] = useState(minRows)

  const handleTextChange = (e) => {
    const newText = e.target.value
    if (newText.length <= maxLength) {
      setText(newText)

      // Auto-resize textarea
      const lineCount = newText.split('\n').length
      const newRows = Math.min(Math.max(lineCount, minRows), maxRows)
      setRows(newRows)
    }
  }

  const handleSubmit = () => {
    if (text.trim()) {
      onTextSubmit(text)
    }
  }

  const handleClear = () => {
    setText('')
    setRows(minRows)
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (file && file.type === 'text/plain') {
      const content = await file.text()
      setText(content.slice(0, maxLength))
    }
  }

  const charCount = text.length
  const isNearLimit = charCount > maxLength * 0.9

  return (
    <div className="w-full">
      {/* Textarea */}
      <div className="relative">
        <div className="absolute top-3 left-3 text-gray-400">
          <Type className="w-5 h-5" />
        </div>

        <textarea
          value={text}
          onChange={handleTextChange}
          rows={rows}
          placeholder={placeholder}
          className={clsx(
            'w-full pl-12 pr-12 py-3 border rounded-lg resize-none',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
            'transition-all duration-200',
            text ? 'border-primary-300' : 'border-gray-300'
          )}
        />

        {text && (
          <button
            onClick={handleClear}
            className="absolute top-3 right-3 p-1 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Clear text"
          >
            <X className="w-5 h-5 text-gray-600" />
          </button>
        )}
      </div>

      {/* Character count and file upload */}
      <div className="flex justify-between items-center mt-2">
        {showCharCount && (
          <span
            className={clsx(
              'text-sm',
              isNearLimit ? 'text-red-600' : 'text-gray-500'
            )}
          >
            {charCount} / {maxLength} characters
          </span>
        )}

        {acceptFiles && (
          <label className="flex items-center space-x-2 cursor-pointer text-sm text-primary-600 hover:text-primary-700">
            <FileText className="w-4 h-4" />
            <span>Upload .txt file</span>
            <input
              type="file"
              accept=".txt"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
        )}
      </div>

      {/* Submit button */}
      <button
        onClick={handleSubmit}
        disabled={!text.trim()}
        className="btn btn-primary w-full mt-4 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Analyze Text
      </button>

      {/* Example prompts */}
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <p className="text-sm font-medium text-gray-700 mb-2">Example texts:</p>
        <div className="space-y-2">
          <button
            onClick={() => setText("This movie was absolutely fantastic! I loved every minute of it.")}
            className="text-xs text-left text-primary-600 hover:text-primary-700 hover:underline block"
          >
            ➤ Positive movie review
          </button>
          <button
            onClick={() => setText("The product quality is terrible. Very disappointed with this purchase.")}
            className="text-xs text-left text-primary-600 hover:text-primary-700 hover:underline block"
          >
            ➤ Negative product review
          </button>
          <button
            onClick={() => setText("The service was okay. Nothing special but not bad either.")}
            className="text-xs text-left text-primary-600 hover:text-primary-700 hover:underline block"
          >
            ➤ Neutral feedback
          </button>
        </div>
      </div>
    </div>
  )
}

export default TextInput
