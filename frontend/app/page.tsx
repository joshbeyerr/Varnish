'use client'

import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState<string | null>(null)
  const [processedImageUrl, setProcessedImageUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [backendStatus, setBackendStatus] = useState<string>('checking...')
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Check backend health
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/health`)
        setBackendStatus('connected ✅')
      } catch (err) {
        setBackendStatus('disconnected ❌')
        setError('Backend server is not running. Please start it with: python backend/main.py')
      }
    }
    checkBackend()
  }, [])

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setUploadResult(null)
      setError(null)
    }
  }

  // Handle file upload
  const handleUpload = async () => {
    if (!selectedFile) return

    setUploading(true)
    setError(null)
    setUploadResult(null)
    setProcessedImageUrl(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setUploadResult(response.data.message)
      // Set the processed image URL for display
      setProcessedImageUrl(`${API_BASE_URL}${response.data.download_url}`)
      setSelectedFile(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    } catch (err) {
      setError('Failed to upload image. Please try again.')
      console.error('Upload error:', err)
    } finally {
      setUploading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-purple-900">
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-md border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
                <img 
                  src="/images/logo-word.png" 
                  alt="Varnish" 
                  className="h-12 w-auto"
                />
              <p className="text-gray-300 mt-1">Protecting Creative Work from AI Training</p>
            </div>
            <div className="text-sm text-gray-300">
              Status: <span className="font-mono text-green-400">{backendStatus}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-6xl font-bold text-white mb-6">
            Take Back Your <span style={{color: '#d069A9'}}>Art</span>
          </h2>
          <p className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
            Protect your creative work from unauthorized AI training and data scraping. 
            Our platform helps artists and creatives maintain control over their intellectual property.
          </p>

          {/* Upload Section */}
          <div className="relative max-w-2xl mx-auto">
            {/* Liquid Glass Background */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-white/5 rounded-3xl blur-xl"></div>
            <div className="relative bg-white/10 backdrop-blur-2xl rounded-3xl shadow-2xl p-10 border border-white/20">
              <h3 className="text-2xl font-semibold text-white mb-8 text-center">
                Upload Your Artwork
              </h3>
              
              {/* Error Display */}
              {error && (
                <div className="mb-6 p-5 bg-red-500/20 backdrop-blur-sm border border-red-500/30 text-red-200 rounded-2xl">
                  <p className="font-semibold">Error:</p>
                  <p>{error}</p>
                </div>
              )}

              {/* Success Display */}
              {uploadResult && (
                <div className="mb-6 p-5 bg-green-500/20 backdrop-blur-sm border border-green-500/30 text-green-200 rounded-2xl">
                  <p className="font-semibold">Success!</p>
                  <p>{uploadResult}</p>
                </div>
              )}

              {/* File Upload */}
              <div className="space-y-8">
                <div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer group relative flex flex-col items-center justify-center w-full h-40 border-2 border-white/30 border-dashed rounded-2xl bg-white/10 backdrop-blur-sm hover:bg-white/20 hover:border-white/50 transition-all duration-500"
                  >
                    {/* Liquid Glass Effect */}
                    <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                    <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                    
                    <div className="flex flex-col items-center justify-center pt-5 pb-6 relative z-10">
                      <svg className="w-12 h-12 mb-4 text-white/70 group-hover:text-white transition-colors duration-300" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
                      </svg>
                      <p className="mb-2 text-sm text-white/90 group-hover:text-white transition-colors duration-300">
                        <span className="font-semibold">Click to upload</span> your artwork
                      </p>
                      <p className="text-xs text-white/60">PNG, JPG, GIF up to 10MB</p>
                    </div>
                  </label>
                </div>

                {selectedFile && (
                  <div className="p-5 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl">
                    <p className="text-sm text-white/90">
                      <span className="font-semibold">Selected:</span> <span className="text-cyan-300">{selectedFile.name}</span>
                    </p>
                    <p className="text-xs text-white/70">
                      Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                )}

                <button
                  onClick={handleUpload}
                  disabled={!selectedFile || uploading}
                  className="group relative w-full px-8 py-5 bg-white/20 backdrop-blur-sm text-white font-bold rounded-2xl border border-white/30 hover:bg-white/30 hover:border-white/50 focus:outline-none focus:ring-2 focus:ring-white/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-500 transform hover:scale-105 hover:shadow-2xl"
                >
                  {/* Liquid Glass Effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                  <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                  
                  {uploading ? (
                    <div className="flex items-center justify-center relative z-10">
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Processing...
                    </div>
                  ) : (
                    <span className="relative z-10 flex items-center justify-center">
                      <img src="/images/white-icon.png" alt="Protect" className="w-5 h-5 mr-2" />
                      Protect My Artwork
                    </span>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Processed Image Display */}
          {processedImageUrl && (
            <div className="mt-12 max-w-2xl mx-auto">
              <div className="relative">
                {/* Liquid Glass Background */}
                <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-white/5 rounded-3xl blur-xl"></div>
                <div className="relative bg-white/10 backdrop-blur-2xl rounded-3xl shadow-2xl p-8 border border-white/20">
                  <h4 className="text-lg font-semibold text-white mb-6 text-center">
                    🛡️ Your Protected Artwork
                  </h4>
                  <div className="flex justify-center">
                    <div className="relative group">
                      <img
                        src={processedImageUrl}
                        alt="Processed artwork"
                        className="max-w-full h-auto rounded-xl shadow-2xl border-2 border-white/30 hover:border-white/50 transition-all duration-300"
                        style={{ maxHeight: '500px' }}
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        <div className="absolute bottom-4 left-4 right-4">
                          <p className="text-white text-sm font-medium">
                            Protected with Blacklight
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="mt-6 text-center">
                    <a
                      href={processedImageUrl}
                      download
                      className="group relative inline-flex items-center px-6 py-3 bg-white/20 backdrop-blur-sm text-white font-medium rounded-2xl border border-white/30 hover:bg-white/30 hover:border-white/50 transition-all duration-500 transform hover:scale-105"
                    >
                      {/* Liquid Glass Effect */}
                      <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                      <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                      
                      <svg className="w-4 h-4 mr-2 relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                      </svg>
                      <span className="relative z-10">Download Protected Image</span>
                    </a>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>


      {/* Features Section */}
      <section className="py-20 px-4 bg-gradient-to-r from-gray-900/50 to-purple-900/30">
        <div className="max-w-6xl mx-auto">
          <h3 className="text-4xl font-bold text-center text-white mb-16">
            How We <span style={{color: '#d069A9'}}>Protect</span> Your Work
          </h3>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center p-8 bg-gray-800/40 backdrop-blur-sm rounded-2xl border border-gray-700 hover:border-cyan-500/50 transition-all duration-300 hover:transform hover:scale-105">
              <div className="w-20 h-20 bg-gradient-to-br from-pink-300 to-pink-400 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
                </svg>
              </div>
              <h4 className="text-xl font-semibold text-white mb-3">AI Glazing</h4>
              <p className="text-gray-300 leading-relaxed">
                Advanced techniques to make your artwork resistant to AI training models
              </p>
            </div>
            <div className="text-center p-8 bg-gray-800/40 backdrop-blur-sm rounded-2xl border border-gray-700 hover:border-cyan-500/50 transition-all duration-300 hover:transform hover:scale-105">
              <div className="w-20 h-20 bg-gradient-to-br from-pink-300 to-pink-400 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
              </div>
              <h4 className="text-xl font-semibold text-white mb-3">Copyright Protection</h4>
              <p className="text-gray-300 leading-relaxed">
                Maintain your intellectual property rights while sharing your art
              </p>
            </div>
            <div className="text-center p-8 bg-gray-800/40 backdrop-blur-sm rounded-2xl border border-gray-700 hover:border-cyan-500/50 transition-all duration-300 hover:transform hover:scale-105">
              <div className="w-20 h-20 bg-gradient-to-br from-pink-300 to-pink-400 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                </svg>
              </div>
              <h4 className="text-xl font-semibold text-white mb-3">Fast Processing</h4>
              <p className="text-gray-300 leading-relaxed">
                Quick and efficient protection without compromising your artwork quality
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Why We Need This Section */}
      <section className="py-20 px-4 bg-gradient-to-br from-blue-900/30 to-purple-900/30">
        <div className="max-w-6xl mx-auto">
          <h3 className="text-4xl font-bold text-center text-white mb-4">
            Why do we need <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">Blacklight</span>?
          </h3>
          <p className="text-center text-gray-300 mb-16 text-lg">
            Learn how images are collected for training AI and Blacklight's protection strategy.
          </p>
          
          <div className="grid md:grid-cols-2 gap-16 items-start">
            {/* Text Content */}
            <div className="space-y-12">
              <div className="space-y-4">
                <h4 className="text-xl font-semibold text-white">Robots collect billions of images from various sources on the web</h4>
                <p className="text-gray-300 leading-relaxed">
                  AI training systems scrape billions of images from platforms like Pixiv, Etsy, DeviantArt, Shopify, and Instagram. 
                  These robots collect human-created art without permission, feeding them into machine learning models.
                </p>
              </div>
              
              <div className="space-y-4">
                <h4 className="text-xl font-semibold text-white">Over time, more images will be generated by AI</h4>
                <p className="text-gray-300 leading-relaxed">
                  As AI image generators learn from human patterns, they create new images that blend AI and human styles. 
                  This makes it increasingly difficult to distinguish between authentic human art and AI-generated content.
                </p>
              </div>
              
              <div className="space-y-4">
                <h4 className="text-xl font-semibold text-white">Current protection methods are easily bypassed</h4>
                <p className="text-gray-300 leading-relaxed">
                  Traditional watermarks and metadata can be removed or disabled. Blacklight's mission is to confuse AI classification 
                  at the pixel level, making your art resistant to style extraction while maintaining its visual quality.
                </p>
              </div>
            </div>
            
            {/* Visual Flow */}
            <div className="flex flex-col items-center space-y-8">
              {/* Cloud 1: Human Art Collection */}
              <div className="relative">
                <div className="w-64 h-40 bg-white/10 backdrop-blur-sm rounded-3xl p-6 border border-white/20">
                  <div className="grid grid-cols-4 gap-2 mb-4">
                    {[...Array(8)].map((_, i) => (
                      <div key={i} className="h-12 bg-gradient-to-br from-purple-400 to-pink-400 rounded-lg"></div>
                    ))}
                  </div>
                  <div className="flex justify-center items-center space-x-2">
                    <div className="w-6 h-6 bg-gray-600 rounded-full flex items-center justify-center">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    </div>
                    <div className="w-8 h-8 bg-gray-800 rounded-full flex items-center justify-center">
                      <div className="w-4 h-4 bg-white rounded-full"></div>
                    </div>
                    <div className="w-5 h-5 bg-gray-600 rounded-full flex items-center justify-center">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                    </div>
                  </div>
                  <div className="text-center mt-2">
                    <span className="text-white text-sm font-medium">Human Art</span>
                  </div>
                </div>
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="bg-cyan-500 text-white px-3 py-1 rounded-full text-xs font-medium">
                    Step 1: Collection
                  </div>
                </div>
              </div>
              
              {/* Arrow */}
              <div className="flex flex-col items-center">
                <svg className="w-6 h-6 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"/>
                </svg>
              </div>
              
              {/* Cloud 2: AI Mixed Content */}
              <div className="relative">
                <div className="w-64 h-40 bg-white/10 backdrop-blur-sm rounded-3xl p-6 border border-white/20">
                  <div className="grid grid-cols-4 gap-2 mb-4">
                    {[...Array(4)].map((_, i) => (
                      <div key={i} className="h-12 bg-gradient-to-br from-purple-400 to-pink-400 rounded-lg"></div>
                    ))}
                    {[...Array(4)].map((_, i) => (
                      <div key={i+4} className="h-12 bg-gray-600 rounded-lg"></div>
                    ))}
                  </div>
                  <div className="flex justify-center items-center space-x-2">
                    <div className="w-8 h-8 bg-gray-800 rounded-full flex items-center justify-center">
                      <div className="w-4 h-4 bg-white rounded-full"></div>
                    </div>
                  </div>
                  <div className="text-center mt-2">
                    <span className="text-white text-sm font-medium">AI Mixed Content</span>
                  </div>
                </div>
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="bg-purple-500 text-white px-3 py-1 rounded-full text-xs font-medium">
                    Step 2: Learning
                  </div>
                </div>
              </div>
              
              {/* Arrow */}
              <div className="flex flex-col items-center">
                <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"/>
                </svg>
              </div>
              
              {/* Cloud 3: Protected Content */}
              <div className="relative">
                <div className="w-64 h-40 bg-white/10 backdrop-blur-sm rounded-3xl p-6 border border-white/20">
                  <div className="grid grid-cols-4 gap-2 mb-4">
                    {[...Array(4)].map((_, i) => (
                      <div key={i} className="h-12 bg-gray-600 rounded-lg"></div>
                    ))}
                    {[...Array(2)].map((_, i) => (
                      <div key={i+4} className="h-12 bg-gradient-to-br from-purple-400 to-pink-400 rounded-lg relative">
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="w-6 h-6 bg-white/20 rounded-full border border-white/40"></div>
                        </div>
                      </div>
                    ))}
                    {[...Array(2)].map((_, i) => (
                      <div key={i+6} className="h-12 bg-gradient-to-br from-cyan-400 to-blue-400 rounded-lg relative">
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="w-6 h-6 bg-white/20 rounded-full border border-white/40"></div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="flex justify-center items-center space-x-2">
                    <div className="w-8 h-8 bg-gradient-to-br from-cyan-400 to-purple-400 rounded-full flex items-center justify-center">
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
                      </svg>
                    </div>
                  </div>
                  <div className="text-center mt-2">
                    <span className="text-white text-sm font-medium">Protected Art</span>
                  </div>
                </div>
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="bg-gradient-to-r from-cyan-500 to-purple-500 text-white px-3 py-1 rounded-full text-xs font-medium">
                    Step 3: Protection
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-black/50 backdrop-blur-sm border-t border-gray-800 py-16 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h4 className="text-3xl font-bold mb-4">
            Bl<span className="text-cyan-400">a</span>ckl<span className="text-purple-400">i</span>ght
          </h4>
          <p className="text-gray-300 mb-8 text-lg">
            Empowering artists to protect their creative work in the age of AI
          </p>
          <div className="flex justify-center space-x-8 mb-8">
            <div className="w-12 h-12 bg-gray-700 rounded-full flex items-center justify-center hover:bg-cyan-600 transition-colors cursor-pointer">
              <svg className="w-6 h-6 text-gray-300" fill="currentColor" viewBox="0 0 24 24">
                <path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z"/>
              </svg>
            </div>
            <div className="w-12 h-12 bg-gray-700 rounded-full flex items-center justify-center hover:bg-purple-600 transition-colors cursor-pointer">
              <svg className="w-6 h-6 text-gray-300" fill="currentColor" viewBox="0 0 24 24">
                <path d="M22.46 6c-.77.35-1.6.58-2.46.69.88-.53 1.56-1.37 1.88-2.38-.83.5-1.75.85-2.72 1.05C18.37 4.5 17.26 4 16 4c-2.35 0-4.27 1.92-4.27 4.29 0 .34.04.67.11.98C8.28 9.09 5.11 7.38 3 4.79c-.37.63-.58 1.37-.58 2.15 0 1.49.75 2.81 1.91 3.56-.71 0-1.37-.2-1.95-.5v.03c0 2.08 1.48 3.82 3.44 4.21a4.22 4.22 0 0 1-1.93.07 4.28 4.28 0 0 0 4 2.98 8.521 8.521 0 0 1-5.33 1.84c-.34 0-.68-.02-1.02-.06C3.44 20.29 5.7 21 8.12 21 16 21 20.33 14.46 20.33 8.79c0-.19 0-.37-.01-.56.84-.6 1.56-1.36 2.14-2.23z"/>
              </svg>
            </div>
          </div>
          <p className="text-sm text-gray-500">
            © 2024 Blacklight. Protecting creative work worldwide.
          </p>
        </div>
      </footer>
    </main>
  )
}
