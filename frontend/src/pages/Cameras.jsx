import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Camera, Plus, Edit, Settings, MapPin, X } from 'lucide-react';
import StatusDot from '../components/StatusDot';

const MOCK_CAMERAS = [
  { id: 1, name: 'Cam-01: Silk Board Junction', location: 'Silk Board, Bengaluru', lat: 12.9176, lon: 77.6244, status: 'online', violations_today: 23, vehicles_tracked: 1245, fps: 28.5, rtsp_url: 'rtsp://192.168.1.101/stream1' },
  { id: 2, name: 'Cam-02: Electronic City Toll', location: 'Electronic City, Bengaluru', lat: 12.8499, lon: 77.6663, status: 'online', violations_today: 18, vehicles_tracked: 987, fps: 30.1, rtsp_url: 'rtsp://192.168.1.102/stream1' },
  { id: 3, name: 'Cam-03: Indiranagar 100ft Rd', location: 'Indiranagar, Bengaluru', lat: 12.9718, lon: 77.6412, status: 'online', violations_today: 31, vehicles_tracked: 1567, fps: 27.8, rtsp_url: 'rtsp://192.168.1.103/stream1' },
  { id: 4, name: 'Cam-04: Marathahalli Bridge', location: 'Marathahalli, Bengaluru', lat: 12.9592, lon: 77.6974, status: 'offline', violations_today: 0, vehicles_tracked: 0, fps: 0, rtsp_url: 'rtsp://192.168.1.104/stream1' },
  { id: 5, name: 'Cam-05: Hebbal Flyover', location: 'Hebbal, Bengaluru', lat: 13.0354, lon: 77.5988, status: 'online', violations_today: 27, vehicles_tracked: 2134, fps: 29.2, rtsp_url: 'rtsp://192.168.1.105/stream1' },
  { id: 6, name: 'Cam-06: Majestic Circle', location: 'Majestic, Bengaluru', lat: 12.9766, lon: 77.5729, status: 'online', violations_today: 19, vehicles_tracked: 876, fps: 28.9, rtsp_url: 'rtsp://192.168.1.106/stream1' },
  { id: 7, name: 'Cam-07: MG Road Junction', location: 'MG Road, Bengaluru', lat: 12.9738, lon: 77.6119, status: 'online', violations_today: 38, vehicles_tracked: 1890, fps: 26.4, rtsp_url: 'rtsp://192.168.1.107/stream1' },
  { id: 8, name: 'Cam-08: Yeshwanthpur Chowk', location: 'Yeshwanthpur, Bengaluru', lat: 13.0285, lon: 77.5402, status: 'offline', violations_today: 0, vehicles_tracked: 0, fps: 0, rtsp_url: 'rtsp://192.168.1.108/stream1' },
];

export default function Cameras() {
  const [showModal, setShowModal] = useState(false);
  const [newCamera, setNewCamera] = useState({ name: '', rtsp_url: '', lat: '', lon: '', description: '' });

  const handleSave = () => {
    alert(`Camera "${newCamera.name}" would be added. (Mock mode)`);
    setShowModal(false);
    setNewCamera({ name: '', rtsp_url: '', lat: '', lon: '', description: '' });
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <h3 className="section-title"><Camera size={18} /> Camera Network</h3>
        <button className="btn btn-primary" onClick={() => setShowModal(true)}><Plus size={16} /> Add Camera</button>
      </div>

      <div className="grid-3">
        {MOCK_CAMERAS.map((cam, i) => (
          <motion.div key={cam.id} className="card camera-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
            <div className="camera-preview">
              <Camera size={32} />
            </div>
            <div>
              <h4 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.25rem' }}>{cam.name}</h4>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.78rem', color: '#64748b' }}>
                <MapPin size={12} /> {cam.location}
              </div>
            </div>
            <div className="camera-info">
              <StatusDot status={cam.status} />
              <span style={{ fontSize: '0.8rem', color: cam.status === 'online' ? '#10b981' : '#64748b', fontWeight: 600 }}>
                {cam.status === 'online' ? 'Online' : 'Offline'}
              </span>
            </div>
            <div className="camera-stats">
              <span>{cam.violations_today} violations</span>
              <span>{cam.vehicles_tracked.toLocaleString()} tracked</span>
              <span>{cam.fps > 0 ? `${cam.fps} FPS` : '—'}</span>
            </div>
            <div className="camera-actions">
              <button className="btn btn-secondary btn-sm">View</button>
              <button className="btn btn-secondary btn-sm"><Edit size={12} /></button>
              <button className="btn btn-secondary btn-sm"><Settings size={12} /></button>
            </div>
          </motion.div>
        ))}
      </div>

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <motion.div className="modal" onClick={e => e.stopPropagation()} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
            <div className="modal-header">
              <h3 className="modal-title">Add New Camera</h3>
              <button className="modal-close" onClick={() => setShowModal(false)}><X size={18} /></button>
            </div>
            <div className="form-group">
              <label className="form-label">Camera Name</label>
              <input className="form-input" placeholder="e.g. Cam-09: University Road" value={newCamera.name} onChange={e => setNewCamera({ ...newCamera, name: e.target.value })} />
            </div>
            <div className="form-group">
              <label className="form-label">RTSP URL</label>
              <input className="form-input" placeholder="rtsp://192.168.1.109/stream1" value={newCamera.rtsp_url} onChange={e => setNewCamera({ ...newCamera, rtsp_url: e.target.value })} />
            </div>
            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">Latitude</label>
                <input className="form-input" type="number" step="0.0001" placeholder="18.5204" value={newCamera.lat} onChange={e => setNewCamera({ ...newCamera, lat: e.target.value })} />
              </div>
              <div className="form-group">
                <label className="form-label">Longitude</label>
                <input className="form-input" type="number" step="0.0001" placeholder="73.8567" value={newCamera.lon} onChange={e => setNewCamera({ ...newCamera, lon: e.target.value })} />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Description</label>
              <textarea className="form-input" rows={3} placeholder="Camera description..." value={newCamera.description} onChange={e => setNewCamera({ ...newCamera, description: e.target.value })} />
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end', marginTop: '1rem' }}>
              <button className="btn btn-secondary" onClick={() => setShowModal(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={handleSave}>Save Camera</button>
            </div>
          </motion.div>
        </div>
      )}
    </motion.div>
  );
}
