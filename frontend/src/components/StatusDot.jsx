import React from 'react';

const STATUS_COLORS = {
  online: '#10b981',
  offline: '#64748b',
  processing: '#3b82f6',
};

export default function StatusDot({ status = 'offline' }) {
  const color = STATUS_COLORS[status] || STATUS_COLORS.offline;
  const isPulsing = status === 'online' || status === 'processing';
  return (
    <span
      className={`status-dot ${isPulsing ? 'pulse' : ''}`}
      style={{ backgroundColor: color }}
      title={status}
    />
  );
}
