import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Chart from 'react-apexcharts';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const TREND_DATA = {
  helmet: [38, 42, 35, 48, 45, 32, 45],
  speed: [28, 31, 25, 35, 32, 22, 32],
  wrong_side: [12, 15, 18, 14, 11, 8, 12],
  red_light: [22, 18, 25, 20, 28, 15, 25],
  illegal_parking: [8, 10, 12, 9, 7, 5, 8],
};

const CAMERA_LOCATIONS = [
  { name: 'Cam-01: Hinjewadi Phase 1', lat: 18.5912, lon: 73.7388, violations: 23, color: '#3b82f6' },
  { name: 'Cam-02: Wakad Bridge', lat: 18.5989, lon: 73.7603, violations: 18, color: '#10b981' },
  { name: 'Cam-03: FC Road Signal', lat: 18.5285, lon: 73.8418, violations: 31, color: '#ef4444' },
  { name: 'Cam-04: Baner Road', lat: 18.5590, lon: 73.7868, violations: 14, color: '#f59e0b' },
  { name: 'Cam-05: Swargate Chowk', lat: 18.5018, lon: 73.8636, violations: 27, color: '#ef4444' },
  { name: 'Cam-06: Kothrud Depot', lat: 18.5074, lon: 73.8077, violations: 19, color: '#3b82f6' },
  { name: 'Cam-07: MG Road Junction', lat: 18.5168, lon: 73.8770, violations: 38, color: '#ef4444' },
  { name: 'Cam-08: Katraj Tunnel', lat: 18.4520, lon: 73.8648, violations: 9, color: '#10b981' },
];

const TYPE_DIST = [
  { type: 'Helmet', count: 245, color: '#ef4444' },
  { type: 'Speed', count: 189, color: '#f59e0b' },
  { type: 'Red Light', count: 156, color: '#f4364c' },
  { type: 'Wrong Side', count: 98, color: '#8b5cf6' },
  { type: 'Parking', count: 67, color: '#d97706' },
  { type: 'Seatbelt', count: 45, color: '#3b82f6' },
  { type: 'Triple Riding', count: 34, color: '#ec4899' },
  { type: 'Distracted', count: 21, color: '#7c3aed' },
];

const VEHICLE_CLASSES = { Car: 45, Motorcycle: 30, Truck: 10, Bus: 8, 'Auto Rickshaw': 5, Bicycle: 2 };
const SPEED_BUCKETS = [
  { range: '0-20', count: 45 },
  { range: '21-40', count: 120 },
  { range: '41-60', count: 280 },
  { range: '61-80', count: 150 },
  { range: '81-100', count: 45 },
  { range: '100+', count: 15 },
];

const PEAK_HOURS = DAYS.map(() => Array.from({ length: 24 }, () => Math.floor(Math.random() * 15)));

const darkChartBase = { chart: { toolbar: { show: false }, background: 'transparent', fontFamily: 'Inter', animations: { enabled: false } }, theme: { mode: 'dark' }, grid: { borderColor: 'rgba(255,255,255,0.05)', strokeDashArray: 4 }, tooltip: { theme: 'dark' } };

export default function Analytics() {
  const trendOptions = {
    ...darkChartBase,
    chart: { ...darkChartBase.chart, type: 'area' },
    colors: ['#ef4444', '#f59e0b', '#8b5cf6', '#f4364c', '#d97706'],
    stroke: { curve: 'smooth', width: 2 },
    fill: { type: 'gradient', gradient: { opacityFrom: 0.3, opacityTo: 0.05 } },
    xaxis: { categories: DAYS, labels: { style: { colors: '#64748b' } } },
    yaxis: { labels: { style: { colors: '#64748b' } } },
    dataLabels: { enabled: false },
    legend: { labels: { colors: '#94a3b8' }, fontSize: '11px' },
  };

  const trendSeries = Object.entries(TREND_DATA).map(([key, data]) => ({
    name: key.replace('_', ' '),
    data,
  }));

  const barOptions = {
    ...darkChartBase,
    chart: { ...darkChartBase.chart, type: 'bar' },
    colors: TYPE_DIST.map(d => d.color),
    plotOptions: { bar: { horizontal: true, barHeight: '60%', borderRadius: 4, distributed: true } },
    xaxis: { categories: TYPE_DIST.map(d => d.type), labels: { style: { colors: '#64748b' } } },
    yaxis: { labels: { style: { colors: '#94a3b8', fontSize: '11px' } } },
    dataLabels: { enabled: false },
    legend: { show: false },
  };

  const vehicleOptions = {
    ...darkChartBase,
    chart: { ...darkChartBase.chart, type: 'donut' },
    labels: Object.keys(VEHICLE_CLASSES),
    colors: ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#64748b'],
    legend: { position: 'bottom', labels: { colors: '#94a3b8' }, fontSize: '11px' },
    dataLabels: { enabled: false },
    plotOptions: { pie: { donut: { size: '65%' } } },
    stroke: { show: false },
  };

  const speedOptions = {
    ...darkChartBase,
    chart: { ...darkChartBase.chart, type: 'bar' },
    colors: ['#06b6d4'],
    plotOptions: { bar: { columnWidth: '60%', borderRadius: 4 } },
    xaxis: { categories: SPEED_BUCKETS.map(b => b.range + ' km/h'), labels: { style: { colors: '#64748b', fontSize: '10px' } } },
    yaxis: { labels: { style: { colors: '#64748b' } } },
    dataLabels: { enabled: false },
  };

  const getCellColor = (value) => {
    if (value < 3) return 'rgba(59,130,246,0.08)';
    if (value < 6) return 'rgba(59,130,246,0.2)';
    if (value < 9) return 'rgba(245,158,11,0.3)';
    if (value < 12) return 'rgba(239,68,68,0.35)';
    return 'rgba(239,68,68,0.55)';
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <h3 className="section-title">Violation Trends (7 Days)</h3>
        <Chart options={trendOptions} series={trendSeries} type="area" height={300} />
      </div>

      <div className="grid-2" style={{ marginBottom: '1.25rem' }}>
        <div className="card">
          <h3 className="section-title">Violation Locations</h3>
          <div style={{ height: 350, borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
            <MapContainer center={[18.5204, 73.8567]} zoom={12} style={{ height: '100%', width: '100%' }} scrollWheelZoom={false}>
              <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution='&copy; CartoDB' />
              {CAMERA_LOCATIONS.map((cam, i) => (
                <CircleMarker key={i} center={[cam.lat, cam.lon]} radius={Math.max(8, cam.violations / 3)} pathOptions={{ fillColor: cam.color, fillOpacity: 0.7, color: cam.color, weight: 2 }}>
                  <Popup>
                    <div style={{ color: '#1e293b' }}>
                      <strong>{cam.name}</strong><br />
                      Violations: {cam.violations}
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>
        </div>
        <div className="card">
          <h3 className="section-title">Type Distribution</h3>
          <Chart options={barOptions} series={[{ name: 'Violations', data: TYPE_DIST.map(d => d.count) }]} type="bar" height={350} />
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: '1.25rem' }}>
        <div className="card">
          <h3 className="section-title">Vehicle Classification</h3>
          <Chart options={vehicleOptions} series={Object.values(VEHICLE_CLASSES)} type="donut" height={300} />
        </div>
        <div className="card">
          <h3 className="section-title">Speed Distribution</h3>
          <Chart options={speedOptions} series={[{ name: 'Vehicles', data: SPEED_BUCKETS.map(b => b.count) }]} type="bar" height={300} />
        </div>
      </div>

      <div className="card">
        <h3 className="section-title">Peak Hours Matrix</h3>
        <div className="peak-matrix">
          <div className="peak-label"></div>
          {Array.from({ length: 24 }, (_, h) => (
            <div key={h} className="peak-label">{h}</div>
          ))}
          {PEAK_HOURS.map((row, dayIdx) => (
            <React.Fragment key={dayIdx}>
              <div className="peak-label">{DAYS[dayIdx]}</div>
              {row.map((val, hourIdx) => (
                <div key={hourIdx} className="peak-cell" style={{ background: getCellColor(val) }} title={`${DAYS[dayIdx]} ${hourIdx}:00 - ${val} violations`}>
                  {val > 0 ? val : ''}
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
