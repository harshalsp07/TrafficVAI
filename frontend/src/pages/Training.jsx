import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Chart from 'react-apexcharts';
import { Play, Square, Download, Rocket } from 'lucide-react';

const TRAINING_HISTORY = [
  { run_id: 'abc123', model: 'YOLOv12-Nano', dataset: 'vehicle_detection', epochs: 100, best_map: 0.847, status: 'completed', duration: '2h 34m', started_at: '2026-06-14' },
  { run_id: 'def456', model: 'YOLOv12-Small', dataset: 'helmet_detection', epochs: 150, best_map: 0.912, status: 'completed', duration: '4h 12m', started_at: '2026-06-13' },
  { run_id: 'ghi789', model: 'YOLOv12-Nano', dataset: 'plate_detection', epochs: 80, best_map: 0.789, status: 'completed', duration: '1h 45m', started_at: '2026-06-12' },
  { run_id: 'jkl012', model: 'YOLOv12-Medium', dataset: 'vehicle_detection', epochs: 200, best_map: 0.0, status: 'failed', duration: '0h 23m', started_at: '2026-06-11' },
  { run_id: 'mno345', model: 'YOLOv12-Nano', dataset: 'seatbelt_classifier', epochs: 50, best_map: 0.756, status: 'completed', duration: '0h 52m', started_at: '2026-06-10' },
];

const AVAILABLE_MODELS = [
  { name: 'yolov12n_vehicles_v3.pt', map: 0.847, size: '13.2 MB', date: '2026-06-14', active: true },
  { name: 'yolov12s_helmet_v2.pt', map: 0.912, size: '22.8 MB', date: '2026-06-13', active: false },
  { name: 'yolov12n_plates_v1.pt', map: 0.789, size: '14.1 MB', date: '2026-06-12', active: false },
];

const LOSS_CURVE = Array.from({ length: 50 }, (_, i) => +(8 * Math.exp(-i / 12) + 0.3 + Math.random() * 0.3).toFixed(3));
const MAP_CURVE = Array.from({ length: 50 }, (_, i) => +(0.85 * (1 - Math.exp(-i / 15)) + Math.random() * 0.02).toFixed(3));

const darkChartBase = { chart: { toolbar: { show: false }, background: 'transparent', fontFamily: 'Inter' }, theme: { mode: 'dark' }, grid: { borderColor: 'rgba(255,255,255,0.05)', strokeDashArray: 4 }, tooltip: { theme: 'dark' } };

export default function Training() {
  const [config, setConfig] = useState({ model: 'yolov12n.pt', dataset: 'datasets/sample/dataset.yaml', epochs: 100, batchSize: 16, imgSize: 640, lr: 0.001, device: 'cpu' });
  const [isTraining, setIsTraining] = useState(false);
  const [mockEpoch, setMockEpoch] = useState(47);

  const handleStart = () => {
    setIsTraining(true);
    setMockEpoch(0);
    const timer = setInterval(() => {
      setMockEpoch(prev => {
        if (prev >= 50) { clearInterval(timer); return prev; }
        return prev + 1;
      });
    }, 200);
  };

  const lossOptions = {
    ...darkChartBase, chart: { ...darkChartBase.chart, type: 'line' },
    colors: ['#ef4444'], stroke: { curve: 'smooth', width: 2 },
    xaxis: { labels: { show: false } }, yaxis: { labels: { style: { colors: '#64748b' } } },
    dataLabels: { enabled: false },
  };
  const mapOptions = {
    ...darkChartBase, chart: { ...darkChartBase.chart, type: 'line' },
    colors: ['#10b981'], stroke: { curve: 'smooth', width: 2 },
    xaxis: { labels: { show: false } }, yaxis: { labels: { style: { colors: '#64748b' } } },
    dataLabels: { enabled: false },
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
      <div className="grid-2" style={{ marginBottom: '1.25rem' }}>
        <div className="card">
          <h3 className="section-title"><Play size={18} /> Start Training</h3>
          <div className="form-group">
            <label className="form-label">Model</label>
            <select className="form-select" value={config.model} onChange={e => setConfig({ ...config, model: e.target.value })}>
              <option value="yolov12n.pt">YOLOv12-Nano</option>
              <option value="yolov12s.pt">YOLOv12-Small</option>
              <option value="yolov12m.pt">YOLOv12-Medium</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Dataset Path</label>
            <input className="form-input" value={config.dataset} onChange={e => setConfig({ ...config, dataset: e.target.value })} />
          </div>
          <div className="form-group">
            <label className="form-label">Epochs: {config.epochs}</label>
            <input type="range" min={10} max={300} value={config.epochs} onChange={e => setConfig({ ...config, epochs: +e.target.value })} />
          </div>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Batch Size</label>
              <select className="form-select" value={config.batchSize} onChange={e => setConfig({ ...config, batchSize: +e.target.value })}>
                {[4, 8, 16, 32].map(b => <option key={b} value={b}>{b}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Image Size</label>
              <select className="form-select" value={config.imgSize} onChange={e => setConfig({ ...config, imgSize: +e.target.value })}>
                {[320, 416, 512, 640].map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          </div>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Learning Rate</label>
              <input className="form-input" type="number" step="0.0001" value={config.lr} onChange={e => setConfig({ ...config, lr: +e.target.value })} />
            </div>
            <div className="form-group">
              <label className="form-label">Device</label>
              <select className="form-select" value={config.device} onChange={e => setConfig({ ...config, device: e.target.value })}>
                <option value="cpu">CPU</option>
                <option value="0">CUDA:0</option>
              </select>
            </div>
          </div>
          <button className="btn btn-primary" style={{ width: '100%', marginTop: '0.5rem' }} onClick={handleStart} disabled={isTraining}>
            <Play size={16} /> {isTraining ? 'Training...' : 'Start Training'}
          </button>
        </div>

        <div className="card">
          <h3 className="section-title">Training Progress</h3>
          {(isTraining || mockEpoch > 0) ? (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
                <span>Epoch <strong>{Math.min(mockEpoch, 50)}</strong> / 50</span>
                <span style={{ color: '#64748b' }}>ETA: {Math.max(0, (50 - mockEpoch) * 3)}s</span>
              </div>
              <div className="progress-bar" style={{ marginBottom: '1rem' }}>
                <div className="progress-fill" style={{ width: `${(mockEpoch / 50) * 100}%` }} />
              </div>
              <div style={{ marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.78rem', color: '#64748b' }}>Training Loss</span>
                <Chart options={lossOptions} series={[{ name: 'Loss', data: LOSS_CURVE.slice(0, mockEpoch + 1) }]} type="line" height={140} />
              </div>
              <div>
                <span style={{ fontSize: '0.78rem', color: '#64748b' }}>mAP@50</span>
                <Chart options={mapOptions} series={[{ name: 'mAP', data: MAP_CURVE.slice(0, mockEpoch + 1) }]} type="line" height={140} />
              </div>
              {isTraining && (
                <button className="btn btn-danger" style={{ width: '100%', marginTop: '0.75rem' }} onClick={() => setIsTraining(false)}>
                  <Square size={14} /> Stop Training
                </button>
              )}
            </>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 300, color: '#64748b', fontSize: '0.85rem' }}>
              No active training. Click "Start Training" to begin.
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <h3 className="section-title">Training History</h3>
        <div className="table-container">
          <table>
            <thead>
              <tr><th>Run ID</th><th>Model</th><th>Dataset</th><th>Epochs</th><th>Best mAP</th><th>Status</th><th>Duration</th></tr>
            </thead>
            <tbody>
              {TRAINING_HISTORY.map(run => (
                <tr key={run.run_id}>
                  <td style={{ fontFamily: 'monospace', color: '#06b6d4' }}>{run.run_id}</td>
                  <td>{run.model}</td>
                  <td>{run.dataset}</td>
                  <td>{run.epochs}</td>
                  <td>{run.best_map > 0 ? run.best_map.toFixed(3) : '—'}</td>
                  <td><span className={`badge badge-${run.status}`}>{run.status}</span></td>
                  <td>{run.duration}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <h3 className="section-title"><Rocket size={18} /> Model Registry</h3>
        <div className="grid-3">
          {AVAILABLE_MODELS.map((model, i) => (
            <motion.div key={i} className="card" style={model.active ? { borderColor: 'rgba(59,130,246,0.5)' } : {}} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}>
              {model.active && <span className="badge" style={{ background: 'rgba(59,130,246,0.15)', color: '#3b82f6', marginBottom: '0.5rem' }}>ACTIVE</span>}
              <h4 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.5rem' }}>{model.name}</h4>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: '#10b981', marginBottom: '0.25rem' }}>{model.map.toFixed(3)}</div>
              <div style={{ fontSize: '0.78rem', color: '#64748b', marginBottom: '0.25rem' }}>mAP@50</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: '#94a3b8', marginBottom: '0.75rem' }}>
                <span>{model.size}</span>
                <span>{model.date}</span>
              </div>
              <button className="btn btn-primary btn-sm" style={{ width: '100%' }} disabled={model.active}>
                <Rocket size={14} /> {model.active ? 'Deployed' : 'Deploy'}
              </button>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
