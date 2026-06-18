import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

// 14 disease classes (index 0 = background, unused in UI)
const CLASS_COLORS = {
  'Aortic enlargement': '#e6194b', 'Atelectasis': '#3cb44b', 'Calcification': '#ffe119',
  'Cardiomegaly': '#4363d8', 'Consolidation': '#f58231', 'ILD': '#911eb4',
  'Infiltration': '#46f0f0', 'Lung Opacity': '#f032e6', 'Nodule/Mass': '#bcf60c',
  'Other lesion': '#fabebe', 'Pleural effusion': '#008080', 'Pleural thickening': '#9a6324',
  'Pneumothorax': '#e6beff', 'Pulmonary fibrosis': '#800000',
}
const IMG_SIZE = 512

function App() {
  const [imageUrl, setImageUrl] = useState(null)
  const [detections, setDetections] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [threshold, setThreshold] = useState(0.25)
  const [hiddenClasses, setHiddenClasses] = useState(new Set())
  const [filename, setFilename] = useState('')
  const [isDicomFile, setIsDicomFile] = useState(false)
  const [dicomFile, setDicomFile] = useState(null)
  const [serverRendered, setServerRendered] = useState(false)
  const canvasRef = useRef(null)
  const imgRef = useRef(null)

  const renderDicom = async (file, thr) => {
    const fd = new FormData()
    fd.append('file', file)
    const visRes = await fetch(`${API_URL}/predict/visualize?score_threshold=${thr}`, {
      method: 'POST', body: fd,
    })
    const blob = await visRes.blob()
    setImageUrl(URL.createObjectURL(blob))
  }
  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setError(null); setDetections([]); setLoading(true)
    setFilename(file.name)

    const isDicom = /\.(dicom|dcm)$/i.test(file.name)
    setIsDicomFile(isDicom)

    const formData = new FormData()
    formData.append('file', file)
    try {
      // Get detections (works for both image and DICOM)
      const res = await fetch(`${API_URL}/predict?score_threshold=0.05`, {
        method: 'POST', body: formData,
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Request failed')
      }
      const data = await res.json()
      setDetections(data.detections)

     if (isDicom) {
        setDicomFile(file)  // keep the file so we can re-render at new thresholds
        await renderDicom(file, threshold)
        setServerRendered(true)
      } else {
        setImageUrl(URL.createObjectURL(file))
        setServerRendered(false)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const loadSample = async () => {
    setError(null); setDetections([]); setLoading(true)
    setFilename('sample_xray.jpg')
    setIsDicomFile(false); setServerRendered(false)
    try {
      // Fetch the bundled sample image and turn it into a File
      const resp = await fetch('/sample_xray.jpg')
      const blob = await resp.blob()
      const file = new File([blob], 'sample_xray.jpg', { type: 'image/jpeg' })

      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_URL}/predict?score_threshold=0.05`, {
        method: 'POST', body: formData,
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Request failed')
      }
      const data = await res.json()
      setDetections(data.detections)
      setImageUrl(URL.createObjectURL(file))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }
  // Redraw canvas whenever image, detections, threshold, or filters change
  useEffect(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !imageUrl) return
    const ctx = canvas.getContext('2d')
    const draw = () => {
      canvas.width = IMG_SIZE
      canvas.height = IMG_SIZE
      ctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE)
      ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE)
      detections
        .filter(d => d.confidence >= threshold && !hiddenClasses.has(d.label))
        .forEach(d => {
          const [x1, y1, x2, y2] = d.box
          const color = CLASS_COLORS[d.label] || '#ff0000'
          ctx.strokeStyle = color
          ctx.lineWidth = 3
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
          const label = `${d.label} ${d.confidence.toFixed(2)}`
          ctx.font = 'bold 14px sans-serif'
          const tw = ctx.measureText(label).width
          ctx.fillStyle = color
          ctx.fillRect(x1, y1 - 18, tw + 8, 18)
          ctx.fillStyle = '#fff'
          ctx.fillText(label, x1 + 4, y1 - 4)
        })
    }
    if (img.complete) draw()
    else img.onload = draw
  }, [imageUrl, detections, threshold, hiddenClasses])

  useEffect(() => {
    if (serverRendered && dicomFile) {
      renderDicom(dicomFile, threshold)
    }
  }, [threshold])
  const toggleClass = (label) => {
    const next = new Set(hiddenClasses)
    next.has(label) ? next.delete(label) : next.add(label)
    setHiddenClasses(next)
  }

  const visible = detections.filter(d => d.confidence >= threshold && !hiddenClasses.has(d.label))
  const presentClasses = [...new Set(detections.map(d => d.label))]

  

  return (
    <div className="app">
      <header>
        <h1>Chest X-Ray Abnormality Detection</h1>
        <p className="subtitle">AI-assisted detection of 14 thoracic findings · Faster R-CNN + WBF</p>
      </header>

      <div className="disclaimer">
        ⚠️ Educational / research demonstration only. <strong>Not a diagnostic device.</strong> Do not use for medical decisions.
      </div>

      <div className="upload-row">
        <label className="upload-btn">
          Upload X-Ray (PNG / JPG / DICOM)
          <input type="file" accept=".png,.jpg,.jpeg,.dicom,.dcm" onChange={handleUpload} hidden />
        </label>
        <button className="sample-btn" onClick={loadSample}>Try a sample image</button>
        {filename && <span className="filename">{filename}</span>}
      </div>

      {loading && (
        <div className="status loading-state">
          <span className="spinner" />
          Analyzing image…
        </div>
      )}
      {error && <div className="status error">Error: {error}</div>}

      {!imageUrl && !loading && (
        <div className="empty-state">
          <div className="empty-icon">🩻</div>
          <p>Upload a chest X-ray or try the sample to see AI-detected findings.</p>
          <p className="empty-sub">Detects 14 thoracic abnormalities · results in seconds</p>
        </div>
      )}

      {imageUrl && !loading && (
        <div className="results">
          <div className="canvas-wrap">
            {serverRendered ? (
              <img src={imageUrl} alt="result" className="result-canvas" />
            ) : (
              <>
                <img ref={imgRef} src={imageUrl} alt="upload" style={{ display: 'none' }} />
                <canvas ref={canvasRef} className="result-canvas" />
              </>
            )}
          </div>

          <div className="panel">
            <div className="control">
              <label>Confidence threshold: <strong>{threshold.toFixed(2)}</strong></label>
              <input type="range" min="0.05" max="0.95" step="0.05"
                value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} />
            </div>

            <h3>Detections ({visible.length})</h3>
            {visible.length === 0 && <p className="muted">No findings above threshold.</p>}
            <ul className="det-list">
              {visible.map((d, i) => (
                <li key={i}>
                  <span className="dot" style={{ background: CLASS_COLORS[d.label] }} />
                  {d.label} <span className="conf">{(d.confidence * 100).toFixed(0)}%</span>
                </li>
              ))}
            </ul>

            {presentClasses.length > 0 && (
              <>
                <h3>Filter classes</h3>
                <div className="filters">
                  {presentClasses.map(c => (
                    <button key={c}
                      className={hiddenClasses.has(c) ? 'chip off' : 'chip'}
                      onClick={() => toggleClass(c)}
                      style={{ borderColor: CLASS_COLORS[c] }}>
                      {c}
                    </button>
                  ))}
                </div>
                
              </>
            )}
          </div>
        </div>
      )}
      <footer className="footer">
        Faster R-CNN (ResNet-50 FPN) · trained on VinBigData with Weighted Boxes Fusion · TTA inference<br/>
        Educational demonstration · not for clinical use
      </footer>


    </div>
  )
}

export default App