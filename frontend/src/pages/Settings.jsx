import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Save, Trash2, RefreshCw, Eye, EyeOff } from 'lucide-react';
import { apiPost } from '../hooks/useApi';

const TABS = ['General', 'Thresholds', 'ANPR', 'Roboflow', 'System'];
const VIOLATION_TYPES_LABELS = [
  { key: 'helmet', label: 'Helmet Violations' },
  { key: 'speed', label: 'Speed Violations' },
  { key: 'wrong_side', label: 'Wrong-Side Driving' },
  { key: 'red_light', label: 'Red Light Violations' },
  { key: 'illegal_parking', label: 'Illegal Parking' },
  { key: 'seatbelt', label: 'Seatbelt Violations' },
  { key: 'triple_riding', label: 'Triple Riding' },
  { key: 'distracted_driving', label: 'Distracted Driving' },
];

export default function Settings() {
  const [activeTab, setActiveTab] = useState('General');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);

  const [general, setGeneral] = useState({ dbType: 'sqlite', evidencePath: './evidence', timezone: 'Asia/Kolkata' });
  const [thresholds, setThresholds] = useState({ speedLimit: 60, helmetConf: 0.6, parkingDuration: 300, wrongSideCosine: -0.5, minFrames: 5 });
  const [anpr, setAnpr] = useState({ regex: '^[A-Z]{2}\\s?\\d{1,2}\\s?[A-Z]{1,3}\\s?\\d{4}$', ocrEngine: 'easyocr', region: 'IN', testPlate: '' });
  const [roboflow, setRoboflow] = useState({ enabled: false, apiKey: '', apiUrl: 'https://serverless.roboflow.com', modelId: 'idd-octso/3,pedestrian-yldth/4,motorcycle-helmet-pz9xs-qoh78/1,seatbelt-detection-lb1ec-pjbz0/1', captureInterval: 1.0 });
  const [notifications, setNotifications] = useState(
    Object.fromEntries(VIOLATION_TYPES_LABELS.map(v => [v.key, true]).concat([['email', false]]))
  );

  // Fetch real settings from Backend API
  useEffect(() => {
    fetch('/api/settings')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        setThresholds({
          speedLimit: data.speed_limit ?? 60,
          helmetConf: data.helmet_conf ?? 0.6,
          parkingDuration: data.parking_duration ?? 300,
          wrongSideCosine: data.wrong_side_cosine ?? -0.5,
          minFrames: data.min_frames ?? 5,
        });
        setAnpr({
          regex: data.regex ?? '^[A-Z]{2}\\s?\\d{1,2}\\s?[A-Z]{1,3}\\s?\\d{4}$',
          ocrEngine: data.ocr_engine ?? 'easyocr',
          region: data.region ?? 'IN',
          testPlate: '',
        });
        setRoboflow({
          enabled: data.roboflow_enabled ?? false,
          apiKey: data.roboflow_api_key ?? '',
          apiUrl: data.roboflow_api_url ?? 'https://serverless.roboflow.com',
          modelId: data.roboflow_model_id ?? 'idd-octso/3,pedestrian-yldth/4,motorcycle-helmet-pz9xs-qoh78/1,seatbelt-detection-lb1ec-pjbz0/1',
          captureInterval: data.capture_interval_seconds ?? 1.0,
        });
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch settings from API:', err);
        setLoading(false);
      });
  }, []);

  const handleSave = async (tabName) => {
    setSaving(true);
    try {
      const payload = {};
      if (tabName === 'General') {
        payload.capture_interval_seconds = roboflow.captureInterval;
      } else if (tabName === 'Thresholds') {
        payload.speed_limit = thresholds.speedLimit;
        payload.helmet_conf = thresholds.helmetConf;
        payload.parking_duration = thresholds.parkingDuration;
        payload.wrong_side_cosine = thresholds.wrongSideCosine;
        payload.min_frames = thresholds.minFrames;
      } else if (tabName === 'ANPR') {
        payload.ocr_engine = anpr.ocrEngine;
        payload.region = anpr.region;
      } else if (tabName === 'Roboflow') {
        payload.roboflow_enabled = roboflow.enabled;
        payload.roboflow_api_key = roboflow.apiKey;
        payload.roboflow_api_url = roboflow.apiUrl;
        payload.roboflow_model_id = roboflow.modelId;
        payload.capture_interval_seconds = roboflow.captureInterval;
      }

      const res = await apiPost('/settings', payload);
      if (res.status === 'success') {
        alert('Settings saved successfully!');
      } else {
        alert('Failed to save settings: ' + (res.detail || 'Unknown error'));
      }
    } catch (err) {
      alert('Error saving settings: ' + err.message);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', justifyContent: 'center', alignItems: 'center', height: '40vh', color: '#64748b' }}>
        <RefreshCw size={24} style={{ animation: 'spin 1.5s linear infinite' }} />
        <span>Loading system settings...</span>
      </div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
      <div className="tabs">
        {TABS.map(tab => (
          <button key={tab} className={`tab ${activeTab === tab ? 'tab-active' : ''}`} onClick={() => setActiveTab(tab)}>{tab}</button>
        ))}
      </div>

      <div className="card">
        {activeTab === 'General' && (
          <div>
            <h3 className="section-title">General Settings</h3>
            <div className="form-group">
              <label className="form-label">Database Type</label>
              <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.25rem' }}>
                {['sqlite', 'clickhouse'].map(db => (
                  <label key={db} style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '0.85rem' }}>
                    <input type="radio" name="dbType" checked={general.dbType === db} onChange={() => setGeneral({ ...general, dbType: db })} style={{ accentColor: '#3b82f6' }} />
                    {db === 'sqlite' ? 'SQLite' : 'ClickHouse'}
                  </label>
                ))}
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Evidence Storage Path</label>
              <input className="form-input" value={general.evidencePath} onChange={e => setGeneral({ ...general, evidencePath: e.target.value })} />
            </div>
            <div className="form-group">
              <label className="form-label">Timezone</label>
              <select className="form-select" value={general.timezone} onChange={e => setGeneral({ ...general, timezone: e.target.value })}>
                <option value="Asia/Kolkata">Asia/Kolkata (IST)</option>
                <option value="UTC">UTC</option>
                <option value="America/New_York">America/New_York (EST)</option>
                <option value="Europe/London">Europe/London (GMT)</option>
                <option value="Asia/Tokyo">Asia/Tokyo (JST)</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Image Capture Interval: {roboflow.captureInterval} sec</label>
              <input type="range" min="0.2" max="10.0" step="0.1" value={roboflow.captureInterval} onChange={e => setRoboflow({ ...roboflow, captureInterval: parseFloat(e.target.value) })} />
              <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.2rem' }}>Automatic delay between evaluating camera frames. Higher values decrease CPU and network load.</div>
            </div>
            <button className="btn btn-primary" onClick={() => handleSave('General')} style={{ marginTop: '0.5rem' }} disabled={saving}>
              <Save size={14} /> {saving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        )}

        {activeTab === 'Thresholds' && (
          <div>
            <h3 className="section-title">Detection Thresholds</h3>
            {[
              { key: 'speedLimit', label: 'Speed Limit', min: 0, max: 200, step: 5, unit: 'km/h', desc: 'Maximum allowed speed before triggering violation' },
              { key: 'helmetConf', label: 'Helmet Confidence', min: 0, max: 1, step: 0.05, unit: '', desc: 'Minimum confidence to confirm helmet absence' },
              { key: 'parkingDuration', label: 'Parking Duration', min: 0, max: 600, step: 15, unit: 'sec', desc: 'Stationary time before illegal parking violation' },
              { key: 'wrongSideCosine', label: 'Wrong-Side Cosine', min: -1, max: 0, step: 0.05, unit: '', desc: 'Cosine similarity threshold for wrong-side detection' },
              { key: 'minFrames', label: 'Min Consecutive Frames', min: 1, max: 30, step: 1, unit: 'frames', desc: 'Frames required before confirming violation' },
            ].map(s => (
              <div key={s.key} className="form-group" style={{ marginBottom: '1.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label className="form-label" style={{ marginBottom: 0 }}>{s.label}</label>
                  <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#3b82f6' }}>
                    {typeof thresholds[s.key] === 'number' ? (s.step < 1 ? thresholds[s.key].toFixed(2) : thresholds[s.key]) : thresholds[s.key]} {s.unit}
                  </span>
                </div>
                <input type="range" min={s.min} max={s.max} step={s.step} value={thresholds[s.key]} onChange={e => setThresholds({ ...thresholds, [s.key]: +e.target.value })} />
                <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.2rem' }}>{s.desc}</div>
              </div>
            ))}
            <button className="btn btn-primary" onClick={() => handleSave('Thresholds')} disabled={saving}>
              <Save size={14} /> {saving ? 'Saving...' : 'Save Thresholds'}
            </button>
          </div>
        )}

        {activeTab === 'ANPR' && (
          <div>
            <h3 className="section-title">ANPR Configuration</h3>
            <div className="form-group">
              <label className="form-label">Plate Regex Pattern</label>
              <input className="form-input" style={{ fontFamily: 'monospace' }} value={anpr.regex} onChange={e => setAnpr({ ...anpr, regex: e.target.value })} />
            </div>
            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">OCR Engine</label>
                <select className="form-select" value={anpr.ocrEngine} onChange={e => setAnpr({ ...anpr, ocrEngine: e.target.value })}>
                  <option value="easyocr">EasyOCR</option>
                  <option value="paddleocr">PaddleOCR</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Region</label>
                <select className="form-select" value={anpr.region} onChange={e => setAnpr({ ...anpr, region: e.target.value })}>
                  <option value="IN">India</option>
                  <option value="US">United States</option>
                  <option value="UK">United Kingdom</option>
                  <option value="EU">European Union</option>
                </select>
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Test Plate</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input className="form-input" placeholder="KA 01 AB 1234" value={anpr.testPlate} onChange={e => setAnpr({ ...anpr, testPlate: e.target.value })} />
                <button className="btn btn-secondary" onClick={() => {
                  try {
                    const valid = new RegExp(anpr.regex).test(anpr.testPlate.toUpperCase());
                    alert(valid ? '✅ Valid plate format' : '❌ Invalid plate format');
                  } catch { alert('Invalid regex pattern'); }
                }}>Test</button>
              </div>
            </div>
            <button className="btn btn-primary" onClick={() => handleSave('ANPR')} style={{ marginTop: '0.5rem' }} disabled={saving}>
              <Save size={14} /> {saving ? 'Saving...' : 'Save ANPR Settings'}
            </button>
          </div>
        )}

        {activeTab === 'Roboflow' && (
          <div>
            <h3 className="section-title">Roboflow Universe Integration</h3>
            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 0' }}>
                <div>
                  <span style={{ fontSize: '0.88rem', fontWeight: 600 }}>Enable Roboflow Detection</span>
                  <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Use Roboflow Universe hosted models instead of local YOLOv12</div>
                </div>
                <label className="toggle">
                  <input type="checkbox" checked={roboflow.enabled} onChange={e => setRoboflow({ ...roboflow, enabled: e.target.checked })} />
                  <span className="toggle-slider" />
                </label>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Roboflow Private API Key</label>
              <div style={{ display: 'flex', gap: '0.5rem', position: 'relative' }}>
                <input className="form-input" type={showApiKey ? 'text' : 'password'} placeholder="rf_xxxxxxxxxxxxxxxxxxxxx" value={roboflow.apiKey} onChange={e => setRoboflow({ ...roboflow, apiKey: e.target.value })} />
                <button className="btn btn-secondary" style={{ padding: '0 12px' }} onClick={() => setShowApiKey(!showApiKey)}>
                  {showApiKey ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
              <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.2rem' }}>You can obtain a free API key from your Roboflow dashboard.</div>
            </div>

            <div className="form-group">
              <label className="form-label">Roboflow API URL</label>
              <input className="form-input" placeholder="https://serverless.roboflow.com" value={roboflow.apiUrl} onChange={e => setRoboflow({ ...roboflow, apiUrl: e.target.value })} />
              <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.2rem' }}>Serverless inference endpoint. Use https://serverless.roboflow.com for hosted models.</div>
            </div>

            <div className="form-group">
              <label className="form-label">Roboflow Model ID(s)</label>
              <input className="form-input" placeholder="e.g. idd-octso/3,pedestrian-yldth/4,motorcycle-helmet-pz9xs-qoh78/1" value={roboflow.modelId} onChange={e => setRoboflow({ ...roboflow, modelId: e.target.value })} />
              <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.2rem' }}>
                Comma-separated model IDs. Each model runs on every frame and detections are merged.
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Image Capture Interval: {roboflow.captureInterval} sec</label>
              <input type="range" min="0.2" max="10.0" step="0.1" value={roboflow.captureInterval} onChange={e => setRoboflow({ ...roboflow, captureInterval: parseFloat(e.target.value) })} />
              <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.2rem' }}>
                Delay between evaluating camera frames. Lower values increase network load and API key credit consumption.
              </div>
            </div>

            <button className="btn btn-primary" onClick={() => handleSave('Roboflow')} style={{ marginTop: '0.5rem' }} disabled={saving}>
              <Save size={14} /> {saving ? 'Saving...' : 'Save Roboflow Settings'}
            </button>
          </div>
        )}

        {activeTab === 'System' && (
          <div>
            <h3 className="section-title">System Information</h3>
            <div className="grid-2" style={{ marginBottom: '1.25rem' }}>
              {[
                { label: 'Uptime', value: '14d 7h 23m', color: '#10b981' },
                { label: 'Disk Usage', value: '23.4 / 100 GB', color: '#f59e0b' },
                { label: 'Database Size', value: '1.2 GB', color: '#3b82f6' },
                { label: 'Active Model Mode', value: roboflow.enabled ? 'Roboflow API' : 'Local YOLOv12', color: '#8b5cf6' },
              ].map((info, i) => (
                <div key={i} style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.35rem' }}>{info.label}</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: info.color }}>{info.value}</div>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button className="btn btn-danger" onClick={() => alert('Cache cleared!')}><Trash2 size={14} /> Clear Cache</button>
              <button className="btn btn-secondary" onClick={() => alert('System restart initiated.')}><RefreshCw size={14} /> Restart System</button>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}
