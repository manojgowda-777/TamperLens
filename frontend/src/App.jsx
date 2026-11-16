
import { useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:5000'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Handle document selection
  function handleFileChange(event) {
    const selectedFile = event.target.files[0]

    if (!selectedFile) return

    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(selectedFile.type)) {
      setError('Please select a JPG or PNG image.')
      return
    }

    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
    setError('')
  }

  // Send the document to the Flask backend
  async function analyzeDocument() {
    if (!file) {
      setError('Please upload a document first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Document analysis failed.')
      }

      setResult(data)
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot connect to Flask. Please check whether the backend is running.'
          : err.message
      )
    } finally {
      setLoading(false)
    }
  }

  // Clear the selected document and previous result
  function resetAnalysis() {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError('')
  }

  const highlightedImage = result
    ? `${API_URL}${result.highlighted_image_url}`
    : null

  return (
    <div className="app">

      {/* Navigation */}
      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon">TL</div>
          <div>
            <h2>TamperLens</h2>
            <span>AI DOCUMENT FORENSICS</span>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          Analysis System
        </div>
      </nav>

      <main className="main-container">

        {/* Introduction */}
        <section className="hero-section">
          <span className="eyebrow">INTELLIGENT DOCUMENT VERIFICATION</span>

          <h1>
            Verify Documents.
            <span> Detect Tampering.</span>
          </h1>

          <p>
            Upload a document image to analyze its authenticity
            and identify potentially suspicious regions using
            custom-trained CNN and U-Net models.
          </p>
        </section>

        {/* Upload Section */}
        <section className="upload-card">
          <div className="section-heading">
            <div>
              <h2>Document Analysis</h2>
              <p>Upload an image to begin verification.</p>
            </div>

            <span className="step-label">STEP 01</span>
          </div>

          <label className="upload-area">
            <input
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              onChange={handleFileChange}
              hidden
            />

            <div className="upload-icon">↑</div>

            <h3>
              {file ? file.name : 'Choose a document image'}
            </h3>

            <p>
              {file
                ? 'Document selected successfully'
                : 'Click here to browse JPG or PNG files'}
            </p>

            <span className="upload-hint">
              Supported formats: JPG, JPEG, PNG
            </span>
          </label>

          {/* Image Preview */}
          {preview && (
            <div className="preview-section">
              <div className="section-heading">
                <h3>Selected Document</h3>
                <button className="text-button" onClick={resetAnalysis}>
                  Remove
                </button>
              </div>

              <img
                src={preview}
                alt="Selected document preview"
                className="document-preview"
              />
            </div>
          )}

          <button
            className="analyze-button"
            onClick={analyzeDocument}
            disabled={!file || loading}
          >
            {loading ? 'Analyzing Document...' : 'Analyze Document'}
            {!loading && <span>→</span>}
          </button>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </section>

        {/* Results */}
        {result && (
          <section className="results-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">ANALYSIS COMPLETE</span>
                <h2>Document Analysis Results</h2>
              </div>
              <span className="step-label">STEP 02</span>
            </div>

            <div className="final-result-card">
              <span>FINAL CLASSIFICATION</span>
              <h2>{result.final_result}</h2>
              <p>
                This result is based on the combined CNN
                classification and U-Net region analysis.
              </p>
            </div>

            <div className="metrics-grid">
              <div className="metric-card">
                <span>CNN PREDICTION</span>
                <h3>{result.cnn_prediction}</h3>
              </div>

              <div className="metric-card">
                <span>AUTHENTIC SCORE</span>
                <h3>{result.authentic_probability}%</h3>
              </div>

              <div className="metric-card">
                <span>TAMPERED SCORE</span>
                <h3>{result.tampered_probability}%</h3>
              </div>

              <div className="metric-card">
                <span>SUSPICIOUS AREA</span>
                <h3>{result.suspicious_area_percentage}%</h3>
              </div>
            </div>

            <div className="highlight-card">
              <div className="section-heading">
                <div>
                  <h3>Suspicious Region Detection</h3>
                  <p>U-Net highlighted output</p>
                </div>
              </div>

              <img
                src={highlightedImage}
                alt="Document with suspicious regions highlighted"
                className="highlighted-image"
              />

              <a
                href={highlightedImage}
                download="TamperLens_Highlighted.png"
                className="download-button"
                target="_blank"
                rel="noreferrer"
              >
                Download Highlighted Image ↓
              </a>
            </div>

            <button className="new-analysis-button" onClick={resetAnalysis}>
              Analyze Another Document
            </button>
          </section>
        )}

        {/* Model Information */}
        <section className="models-section">
          <div className="model-card">
            <div className="model-icon">CNN</div>
            <div>
              <h3>Document Classification</h3>
              <p>Custom CNN model identifies document authenticity.</p>
            </div>
          </div>

          <div className="model-card">
            <div className="model-icon">U-Net</div>
            <div>
              <h3>Tampering Localization</h3>
              <p>Custom U-Net identifies and highlights suspicious regions.</p>
            </div>
          </div>
        </section>

      </main>

      <footer>
        <p>TamperLens © 2026 | AI-Based Document Tampering Detection</p>
        <span>Results are automated assessments, not definitive proof of forgery.</span>
      </footer>

    </div>
  )
}

export default App
