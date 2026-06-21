import React, { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload as UploadIcon, Image, AlertTriangle, CheckCircle, Loader2, X, Download, Clock } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || '';

const VIOLATION_LABELS = {
  helmet_non_compliance: { label: 'Helmet Violation', color: '#ef4444', bg: 'rgba(239,68,68,0.15)' },
  triple_riding: { label: 'Triple Riding', color: '#f59e0b', bg: 'rgba(245,158,11,0.15)' },
  seatbelt_non_compliance: { label: 'Seatbelt Violation', color: '#3b82f6', bg: 'rgba(59,130,246,0.15)' },
  speed_violation: { label: 'Speed Violation', color: '#3b82f6', bg: 'rgba(59,130,246,0.15)' },
  wrong_side_driving: { label: 'Wrong Side Driving', color: '#8b5cf6', bg: 'rgba(139,92,246,0.15)' },
  red_light_running: { label: 'Red Light Running', color: '#f43e4c', bg: 'rgba(244,62,76,0.15)' },
  stop_line_violation: { label: 'Stop Line Violation', color: '#a855f7', bg: 'rgba(168,85,247,0.15)' },
  illegal_parking: { label: 'Illegal Parking', color: '#22c55e', bg: 'rgba(34,197,94,0.15)' },
};

export default function Upload() {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  }, []);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file (JPEG, PNG, WEBP)');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError('File too large. Maximum 20MB allowed.');
      return;
    }
    setSelectedFile(file);
    setResult(null);
    setError(null);

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [handleFile]);

  const handleChange = useCallback((e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  }, [handleFile]);

  const processImage = async () => {
    if (!selectedFile) return;
    setProcessing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`${API_BASE}/api/inference/image`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to process image. Is the backend running?');
    } finally {
      setProcessing(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const downloadAnnotated = () => {
    if (!result?.annotated_image_url) return;
    const link = document.createElement('a');
    link.href = `${API_BASE}${result.annotated_image_url}`;
    link.download = `annotated_${selectedFile?.name || 'image.jpg'}`;
    link.click();
  };

  return (
    <div className="upload-page">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="upload-header">
          <h2 className="upload-title">Image Violation Detection</h2>
          <p className="upload-subtitle">
            Upload a traffic image to detect violations using AI-powered analysis
          </p>
        </div>

        {!result && (
          <div
            className={`upload-dropzone ${dragActive ? 'dropzone-active' : ''} ${preview ? 'dropzone-has-preview' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => !preview && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              onChange={handleChange}
              style={{ display: 'none' }}
            />

            {preview ? (
              <div className="preview-container">
                <img src={preview} alt="Preview" className="image-preview" />
                <button className="preview-remove" onClick={(e) => { e.stopPropagation(); reset(); }}>
                  <X size={18} />
                </button>
              </div>
            ) : (
              <div className="dropzone-content">
                <div className="dropzone-icon">
                  <UploadIcon size={48} />
                </div>
                <p className="dropzone-text">Drag & drop a traffic image here</p>
                <p className="dropzone-hint">or click to browse files</p>
                <p className="dropzone-formats">JPEG, PNG, WEBP, BMP — Max 20MB</p>
              </div>
            )}
          </div>
        )}

        {preview && !result && (
          <motion.div className="upload-actions" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <button
              className="btn btn-primary btn-lg"
              onClick={processImage}
              disabled={processing}
            >
              {processing ? (
                <>
                  <Loader2 size={20} className="spin" />
                  Processing...
                </>
              ) : (
                <>
                  <AlertTriangle size={20} />
                  Detect Violations
                </>
              )}
            </button>
            <button className="btn btn-ghost" onClick={reset} disabled={processing}>
              Choose Different Image
            </button>
          </motion.div>
        )}

        {processing && (
          <motion.div className="processing-overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div className="processing-card">
              <Loader2 size={48} className="spin" />
              <h3>Analyzing Image...</h3>
              <p>Running 7-stage detection pipeline</p>
              <div className="processing-steps">
                <div className="step active">1. Quality Assessment</div>
                <div className="step active">2. Object Detection</div>
                <div className="step active">3. Object Tracking</div>
                <div className="step active">4. Classification</div>
                <div className="step active">5. Violation Rules</div>
                <div className="step">6. License Plate OCR</div>
                <div className="step">7. Evidence Generation</div>
              </div>
            </div>
          </motion.div>
        )}

        {error && (
          <motion.div className="error-banner" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
            <AlertTriangle size={20} />
            <span>{error}</span>
            <button onClick={() => setError(null)}><X size={16} /></button>
          </motion.div>
        )}

        <AnimatePresence>
          {result && (
            <motion.div
              className="result-container"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <div className="result-header">
                <div className="result-status">
                  {result.total_violations > 0 ? (
                    <AlertTriangle size={24} className="text-warning" />
                  ) : (
                    <CheckCircle size={24} className="text-success" />
                  )}
                  <div>
                    <h3>
                      {result.total_violations > 0
                        ? `${result.total_violations} Violation${result.total_violations !== 1 ? 's' : ''} Detected`
                        : 'No Violations Detected'}
                    </h3>
                    <p className="result-meta">
                      {result.total_detections} object{result.total_detections !== 1 ? 's' : ''} found
                      {' '}&middot;{' '}
                      <Clock size={14} /> {result.processing_time_ms}ms
                      {result.status === 'partial' && ' (partial results)'}
                    </p>
                  </div>
                </div>
                <div className="result-actions">
                  <button className="btn btn-primary" onClick={downloadAnnotated}>
                    <Download size={16} />
                    Download Annotated
                  </button>
                  <button className="btn btn-ghost" onClick={reset}>
                    <UploadIcon size={16} />
                    Upload Another
                  </button>
                </div>
              </div>

              <div className="result-body">
                <div className="result-image-panel">
                  <h4>Annotated Image</h4>
                  <div className="annotated-image-wrapper">
                    <img
                      src={`${API_BASE}${result.annotated_image_url}`}
                      alt="Annotated traffic image"
                      className="annotated-image"
                    />
                  </div>
                </div>

                <div className="result-violations-panel">
                  <h4>Detected Violations</h4>
                  {result.violations.length === 0 ? (
                    <div className="no-violations">
                      <CheckCircle size={40} />
                      <p>No violations found in this image</p>
                      <p className="no-violations-hint">The AI analysis found all vehicles compliant</p>
                    </div>
                  ) : (
                    <div className="violations-list">
                      {result.violations.map((v, i) => {
                        const meta = VIOLATION_LABELS[v.violation_type] || { label: v.violation_type, color: '#94a3b8', bg: 'rgba(148,163,184,0.15)' };
                        return (
                          <motion.div
                            key={i}
                            className="violation-card"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.1 }}
                          >
                            <div className="violation-card-header">
                              <span className="violation-badge" style={{ color: meta.color, background: meta.bg }}>
                                {meta.label}
                              </span>
                              <span className="violation-confidence">
                                {(v.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                            {v.details?.license_plate && (
                              <div className="violation-plate">
                                Plate: <strong>{v.details.license_plate}</strong>
                              </div>
                            )}
                            <div className="violation-bbox">
                              Location: [{v.bbox?.map(b => Math.round(b)).join(', ')}]
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  )}

                  {result.quality && (
                    <div className="quality-info">
                      <h5>Image Quality</h5>
                      <div className="quality-stats">
                        {result.quality.blur_score != null && (
                          <span>Blur: {(result.quality.blur_score * 100).toFixed(0)}%</span>
                        )}
                        {result.quality.brightness != null && (
                          <span>Brightness: {(result.quality.brightness * 100).toFixed(0)}%</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {!result && !processing && (
          <div className="upload-info-cards">
            <div className="info-card">
              <AlertTriangle size={24} className="info-card-icon" />
              <h4>8 Violation Types</h4>
              <p>Helmet, triple riding, seatbelt, speed, wrong-side, red-light, stop-line, illegal parking</p>
            </div>
            <div className="info-card">
              <Image size={24} className="info-card-icon" />
              <h4>AI-Powered Detection</h4>
              <p>YOLOv12 + ByteTrack multi-object tracking with micro-classifiers</p>
            </div>
            <div className="info-card">
              <CheckCircle size={24} className="info-card-icon" />
              <h4>Annotated Results</h4>
              <p>Get bounding boxes, confidence scores, and license plate recognition</p>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}
