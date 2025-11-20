import React, { useCallback, useState, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { Mic, Upload, X, Play, Pause, Volume2 } from 'lucide-react'
import clsx from 'clsx'

const AudioUpload = ({
  onFileSelect,
  maxSize = 50 * 1024 * 1024, // 50MB default
  selectedFile,
  onClear,
  enableRecording = false,
}) => {
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const audioRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles && acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0])
      }
    },
    [onFileSelect]
  )

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    },
    maxSize,
    multiple: false,
  })

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const file = new File([blob], 'recording.webm', { type: 'audio/webm' })
        onFileSelect(file)
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error('Error accessing microphone:', error)
      alert('Could not access microphone. Please check permissions.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  // Playback functions
  const togglePlayback = () => {
    if (!audioRef.current) return

    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
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

          <div className="flex flex-col items-center space-y-4">
            <Upload
              className={clsx(
                'w-12 h-12',
                isDragActive ? 'text-primary-500' : 'text-gray-400'
              )}
            />

            {isDragActive ? (
              <p className="text-primary-600 font-medium">Drop the audio file here</p>
            ) : (
              <>
                <p className="text-gray-700 font-medium">
                  Drag & drop an audio file here, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  Supported: WAV, MP3, FLAC, OGG, M4A
                </p>
                <p className="text-sm text-gray-500">
                  Max file size: {formatFileSize(maxSize)}
                </p>
              </>
            )}
          </div>
        </div>
      )}

      {/* Recording option */}
      {enableRecording && !selectedFile && (
        <div className="mt-4">
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-2">Or record audio directly:</p>
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={clsx(
                'inline-flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all',
                isRecording
                  ? 'bg-red-600 text-white hover:bg-red-700'
                  : 'bg-primary-600 text-white hover:bg-primary-700'
              )}
            >
              <Mic className="w-5 h-5" />
              <span>{isRecording ? 'Stop Recording' : 'Start Recording'}</span>
            </button>
            {isRecording && (
              <p className="text-sm text-red-600 mt-2 animate-pulse">
                🔴 Recording in progress...
              </p>
            )}
          </div>
        </div>
      )}

      {/* Selected File Display */}
      {selectedFile && (
        <div className="border-2 border-primary-200 bg-primary-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                <Volume2 className="w-5 h-5 text-primary-600" />
              </div>
              <div>
                <p className="font-medium text-gray-900">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">
                  {formatFileSize(selectedFile.size)}
                  {duration > 0 && ` • ${formatDuration(duration)}`}
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

          {/* Audio Player */}
          <div className="space-y-3">
            <audio
              ref={audioRef}
              src={selectedFile ? URL.createObjectURL(selectedFile) : ''}
              onLoadedMetadata={handleLoadedMetadata}
              onEnded={() => setIsPlaying(false)}
              className="hidden"
            />

            <button
              onClick={togglePlayback}
              className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              {isPlaying ? (
                <>
                  <Pause className="w-4 h-4" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Play Preview</span>
                </>
              )}
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

export default AudioUpload
