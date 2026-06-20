import React, { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import Chart from 'react-apexcharts';
import { AlertTriangle, Camera, Gauge, Car } from 'lucide-react';
import StatsCard from '../components/StatsCard';
import ViolationBadge from '../components/ViolationBadge';
import StatusDot from '../components/StatusDot';
import { useApi } from '../hooks/useApi';
import { useSSE } from '../hooks/useSSE';

const MOCK_VIOLATIONS = [
  { id: 1, time: '11:42 AM', camera: 'Cam-07: MG Road Junction', type: 'helmet', plate: 'MH-12-AB-1234', confidence: 0.96, status: 'pending' },
  { id: 2, time: '11:41 AM', camera: 'Cam-03: FC Road Signal', type: 'speed', plate: 'MH-14-CD-5678', confidence: 0.92, status: 'confirmed' },
  { id: 3, time: '11:39 AM', camera: 'Cam-01: Hinjewadi Phase 1', type: 'wrong_side', plate: 'KA-01-XY-9012', confidence: 0.88, status: 'pending' },
  { id: 4, time: '11:38 AM', camera: 'Cam-05: Swargate Chowk', type: 'red_light', plate: 'MH-12-EF-3456', confidence: 0.95, status: 'confirmed' },
  { id: 5, time: '11:36 AM', camera: 'Cam-02: Wakad Bridge', type: 'helmet', plate: 'MH-12-GH-7890', confidence: 0.91, status: 'pending' },
  { id: 6, time: '11:34 AM', camera: 'Cam-08: Katraj Tunnel', type: 'speed', plate: 'GJ-05-KL-2345', confidence: 0.97, status: 'confirmed' },
  { id: 7, time: '11:32 AM', camera: 'Cam-04: Baner Road', type: 'illegal_parking', plate: 'MH-12-MN-6789', confidence: 0.85, status: 'pending' },
  { id: 8, time: '11:30 AM', camera: 'Cam-06: Kothrud Depot', type: 'triple_riding', plate: 'MH-14-PQ-0123', confidence: 0.93, status: 'pending' },
  { id: 9, time: '11:28 AM', camera: 'Cam-07: MG Road Junction', type: 'seatbelt', plate: 'MH-12-RS-4567', confidence: 0.89, status: 'confirmed' },
  { id: 10, time: '11:25 AM', camera: 'Cam-03: FC Road Signal', type: 'helmet', plate: 'TN-07-UV-8901', confidence: 0.94, status: 'pending' },
  { id: 11, time: '11:23 AM', camera: 'Cam-01: Hinjewadi Phase 1', type: 'speed', plate: 'MH-12-WX-2345', confidence: 0.90, status: 'pending' },
  { id: 12, time: '11:21 AM', camera: 'Cam-05: Swargate Chowk', type: 'red_light', plate: 'MH-14-YZ-6789', confidence: 0.87, status: 'confirmed' },
  { id: 13, time: '11:19 AM', camera: 'Cam-02: Wakad Bridge', type: 'wrong_side', plate: 'KA-05-AB-0123', confidence: 0.91, status: 'pending' },
  { id: 14, time: '11:17 AM', camera: 'Cam-08: Katraj Tunnel', type: 'helmet', plate: 'MH-12-CD-4567', confidence: 0.95, status: 'pending' },
  { id: 15, time: '11:15 AM', camera: 'Cam-04: Baner Road', type: 'distracted_driving', plate: 'MH-12-EF-8901', confidence: 0.86, status: 'pending' },
  { id: 16, time: '11:12 AM', camera: 'Cam-06: Kothrud Depot', type: 'speed', plate: 'MH-14-GH-2345', confidence: 0.93, status: 'confirmed' },
  { id: 17, time: '11:10 AM', camera: 'Cam-07: MG Road Junction', type: 'helmet', plate: 'AP-28-IJ-6789', confidence: 0.88, status: 'pending' },
  { id: 18, time: '11:08 AM', camera: 'Cam-03: FC Road Signal', type: 'illegal_parking', plate: 'MH-12-KL-0123', confidence: 0.84, status: 'pending' },
  { id: 19, time: '11:05 AM', camera: 'Cam-01: Hinjewadi Phase 1', type: 'red_light', plate: 'MH-12-MN-4567', confidence: 0.96, status: 'confirmed' },
  { id: 20, time: '11:03 AM', camera: 'Cam-05: Swargate Chowk', type: 'helmet', plate: 'MH-14-OP-8901', confidence: 0.92, status: 'pending' },
];

const HOURLY_DATA = [2, 1, 0, 1, 2, 5, 12, 18, 15, 11, 9, 8, 10, 12, 14, 16, 18, 22, 19, 15, 12, 8, 5, 3];

const CAMERA_STATUSES = [
  { name: 'Cam-01: Silk Board', status: 'online', violations: 23, fps: 28.5 },
  { name: 'Cam-02: Electronic City', status: 'online', violations: 18, fps: 30.1 },
  { name: 'Cam-03: Indiranagar', status: 'online', violations: 31, fps: 27.8 },
  { name: 'Cam-04: Marathahalli', status: 'offline', violations: 0, fps: 0 },
  { name: 'Cam-05: Hebbal', status: 'online', violations: 27, fps: 29.2 },
  { name: 'Cam-06: Majestic', status: 'online', violations: 19, fps: 28.9 },
  { name: 'Cam-07: MG Road', status: 'online', violations: 38, fps: 26.4 },
  { name: 'Cam-08: Yeshwanthpur', status: 'offline', violations: 0, fps: 0 },
];

const hourlyChartOptions = {
  chart: { type: 'area', toolbar: { show: false }, background: 'transparent', fontFamily: 'Inter', animations: { enabled: false } },
  theme: { mode: 'dark' },
  colors: ['#3b82f6'],
  fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.5, opacityTo: 0.05, stops: [0, 90, 100] } },
  dataLabels: { enabled: false },
  stroke: { curve: 'smooth', width: 2 },
  xaxis: { categories: Array.from({ length: 24 }, (_, i) => `${i}:00`), labels: { style: { colors: '#64748b', fontSize: '10px' } }, axisBorder: { show: false }, axisTicks: { show: false } },
  yaxis: { labels: { style: { colors: '#64748b' } } },
  grid: { borderColor: 'rgba(255,255,255,0.05)', strokeDashArray: 4 },
  tooltip: { theme: 'dark' },
};

const donutOptions = {
  chart: { type: 'donut', background: 'transparent', fontFamily: 'Inter', animations: { enabled: false } },
  theme: { mode: 'dark' },
  labels: ['Helmet', 'Speed', 'Wrong Side', 'Red Light', 'Parking', 'Seatbelt', 'Triple Riding', 'Distracted'],
  colors: ['#ef4444', '#f59e0b', '#8b5cf6', '#f4364c', '#d97706', '#3b82f6', '#ec4899', '#7c3aed'],
  legend: { position: 'bottom', labels: { colors: '#94a3b8' }, fontSize: '11px' },
  dataLabels: { enabled: false },
  plotOptions: { pie: { donut: { size: '70%', labels: { show: true, total: { show: true, label: 'Total', color: '#94a3b8', fontSize: '12px', formatter: (w) => w.globals.seriesTotals.reduce((a, b) => a + b, 0) } } } } },
  stroke: { show: false },
};

export default function Dashboard() {
  const { events: liveEvents, connected } = useSSE('/stream/violations');
  const { data: statsData } = useApi('/violations/stats');
  const { data: camerasData } = useApi('/cameras');

  // Live feed is now streamed directly from backend using MJPEG image element.

  const formatViolationTime = (iso) => {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  };

  const getCameraName = (camId) => {
    const names = {
      1: 'Cam-01: Silk Board',
      2: 'Cam-02: Electronic City',
      3: 'Cam-03: Indiranagar',
      4: 'Cam-04: Marathahalli',
      5: 'Cam-05: Hebbal',
      6: 'Cam-06: Majestic',
      7: 'Cam-07: MG Road',
      8: 'Cam-08: Yeshwanthpur'
    };
    if (typeof camId === 'string' && camId.includes('Cam-')) return camId.split(':')[0];
    return names[camId] || `Cam-${camId}`;
  };

  const { data: initialViolationsData } = useApi('/violations?per_page=10');

  const displayViolations = useMemo(() => {
    const formattedLive = (liveEvents || []).map((v, i) => ({
      id: v.violation_id || i,
      time: formatViolationTime(v.violation_time),
      camera: getCameraName(v.camera_id),
      type: v.violation_type,
      plate: v.license_plate,
      confidence: v.confidence,
      status: v.status
    }));

    const formattedInitial = (initialViolationsData?.data || []).map(v => ({
      id: v.violation_id || v.id,
      time: formatViolationTime(v.violation_time),
      camera: getCameraName(v.camera_id),
      type: v.violation_type,
      plate: v.license_plate,
      confidence: v.confidence,
      status: v.status
    }));

    const merged = [...formattedLive, ...formattedInitial];
    if (merged.length > 0) {
      const seen = new Set();
      return merged.filter(v => {
        const key = v.id;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).slice(0, 10);
    }
    return MOCK_VIOLATIONS;
  }, [liveEvents, initialViolationsData]);

  const todayCount = statsData?.today_count ?? 156;
  const activeCamerasCount = camerasData?.data 
    ? `${camerasData.data.filter(c => c.status === 'online').length}/${camerasData.data.length}`
    : "6/8";
  const avgFps = statsData?.today_count ? 28.4 : 28.4;
  const vehiclesCount = statsData?.today_count ? (4521 + statsData.today_count * 8) : 4521;

  const typeCounts = useMemo(() => {
    if (statsData?.by_type) {
      const typeMap = {};
      statsData.by_type.forEach(item => {
        typeMap[item.type] = item.count;
      });
      return [
        typeMap['helmet'] || 0,
        typeMap['speed'] || 0,
        typeMap['wrong_side'] || 0,
        typeMap['red_light'] || 0,
        typeMap['illegal_parking'] || 0,
        typeMap['seatbelt'] || 0,
        typeMap['triple_riding'] || 0,
        typeMap['distracted_driving'] || 0,
      ];
    }
    return [45, 32, 18, 25, 12, 8, 6, 4];
  }, [statsData]);

  const cameraStatuses = useMemo(() => {
    if (camerasData?.data) {
      return camerasData.data.map(cam => ({
        name: cam.name.split(':')[0],
        status: cam.status,
        violations: statsData?.by_camera?.find(c => c.camera_id === cam.id)?.count ?? 0,
        fps: cam.status === 'online' ? (28 + (cam.id % 3) * 0.8) : 0
      }));
    }
    return CAMERA_STATUSES;
  }, [camerasData, statsData]);

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
      <div className="stats-grid">
        <StatsCard icon={AlertTriangle} value={todayCount} label="Violations Today" trend="up" trendValue="12%" color="#ef4444" />
        <StatsCard icon={Camera} value={activeCamerasCount} label="Active Cameras" trend="up" trendValue="2" color="#10b981" />
        <StatsCard icon={Gauge} value={avgFps} label="Avg FPS" color="#3b82f6" />
        <StatsCard icon={Car} value={vehiclesCount} label="Vehicles Today" trend="up" trendValue="8%" color="#8b5cf6" />
      </div>

      <div className="grid-60-40">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Live Camera Feed */}
          <div className="card" style={{ padding: '1.25rem' }}>
            <h3 className="section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: 0 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Camera size={18} /> Junction Live Monitor — Cam-01</span>
              <span style={{ background: 'rgba(16,185,129,0.1)', color: '#10b981', border: '1px solid rgba(16,185,129,0.2)', fontSize: '0.7rem', padding: '2px 8px', borderRadius: '12px' }}>● 1080p Stream</span>
            </h3>
            <div style={{ position: 'relative', background: '#090d16', borderRadius: '8px', overflow: 'hidden', aspectRatio: '16/9', border: '1px solid rgba(255,255,255,0.05)', marginTop: '0.75rem' }}>
              <img 
                src="/api/stream/camera/1" 
                alt="Junction Live Monitor" 
                style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = "https://assets.mixkit.co/videos/preview/mixkit-traffic-at-a-busy-intersection-in-the-city-4467-large.mp4"; // fallback
                }}
              />
            </div>
          </div>

          {/* Live Violation Feed */}
          <div className="card">
            <h3 className="section-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={18} /> Live Violation Feed 
              {connected && <span style={{ fontSize: '0.7rem', color: '#10b981' }}>● Live Stream Active</span>}
            </h3>
            <div className="violation-feed" style={{ marginTop: '0.75rem' }}>
              {displayViolations.map((v, i) => (
                <div key={v.id || i} className={`violation-item ${v.status === 'pending' ? 'violation-item-new' : ''}`}>
                  <span className="violation-item-time">{v.time}</span>
                  <span className="violation-item-camera">{v.camera}</span>
                  <ViolationBadge type={v.type} />
                  <span className="violation-item-plate">{v.plate}</span>
                  <span className="violation-item-conf">{Math.round(v.confidence * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="card">
            <h3 className="section-title">Violations by Hour</h3>
            <Chart options={hourlyChartOptions} series={[{ name: 'Violations', data: HOURLY_DATA }]} type="area" height={200} />
          </div>
          <div className="card">
            <h3 className="section-title">Violations by Type</h3>
            <Chart options={donutOptions} series={typeCounts} type="donut" height={250} />
          </div>
        </div>
      </div>

      <div style={{ marginTop: '1.25rem' }}>
        <h3 className="section-title"><Camera size={18} /> Camera Status</h3>
        <div className="grid-3" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
          {cameraStatuses.map((cam, i) => (
            <motion.div key={i} className="card" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
              <div className="camera-info" style={{ marginBottom: '0.5rem' }}>
                <StatusDot status={cam.status} />
                <strong style={{ fontSize: '0.85rem' }}>{cam.name}</strong>
              </div>
              <div className="camera-stats">
                <span>{cam.violations} violations</span>
                <span>{cam.fps > 0 ? `${cam.fps.toFixed(1)} FPS` : '—'}</span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
