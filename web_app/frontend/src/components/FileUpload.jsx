import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, X, File } from 'lucide-react'
import clsx from 'clsx'

const FileUpload = ({
  onFileSelect,
  acceptedTypes,
  maxSize = 10 * 1024 * 1024, // 10MB default
  multiple = false,
  selectedFile,
  onClear,
}) => {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles && acceptedFiles.length > 0) {
        if (multiple) {
          onFileSelect(acceptedFiles)
        } else {
          onFileSelect(acceptedFiles[0])
        }
      }
    },
    [onFileSelect, multiple]
  )

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: acceptedTypes,
    maxSize,
    multiple,
  })

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className="w-full">
      {/* Dropzone */}
      {!selectedFile && (
        <div
          {...getRootProps()}
          className={clsx(
            'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all',
            isDragActive
              ? 'border-primary-500 bg-primary-50'
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          )}
        >
          <input {...getInputProps()} />

          <Upload
            className={clsx(
              'w-12 h-12 mx-auto mb-4',
              isDragActive ? 'text-primary-500' : 'text-gray-400'
            )}
          />

          {isDragActive ? (
            <p className="text-primary-600 font-medium">Drop the file here</p>
          ) : (
            <>
              <p className="text-gray-700 font-medium mb-2">
                Drag & drop a file here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Max file size: {formatFileSize(maxSize)}
              </p>
            </>
          )}
        </div>
      )}

      {/* Selected File Display */}
      {selectedFile && (
        <div className="border-2 border-primary-200 bg-primary-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                <File className="w-5 h-5 text-primary-600" />
              </div>
              <div>
                <p className="font-medium text-gray-900">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">
                  {formatFileSize(selectedFile.size)}
                </p>
              </div>
            </div>

            <button
              onClick={onClear}
              className="p-2 hover:bg-primary-100 rounded-lg transition-colors"
              aria-label="Clear file"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </div>
      )}

      {/* Error Messages */}
      {fileRejections.length > 0 && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm font-medium text-red-800 mb-2">
            File rejected:
          </p>
          <ul className="list-disc list-inside text-sm text-red-700">
            {fileRejections.map(({ file, errors }) => (
              <li key={file.path}>
                {file.path} - {errors.map((e) => e.message).join(', ')}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default FileUpload
