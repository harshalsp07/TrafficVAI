import React from 'react';

const TYPE_COLORS = {
  helmet: { bg: '#ef444420', color: '#ef4444', label: 'Helmet' },
  speed: { bg: '#f59e0b20', color: '#f59e0b', label: 'Speed' },
  wrong_side: { bg: '#8b5cf620', color: '#8b5cf6', label: 'Wrong Side' },
  red_light: { bg: '#f4364c20', color: '#f4364c', label: 'Red Light' },
  illegal_parking: { bg: '#d9770620', color: '#d97706', label: 'Parking' },
  seatbelt: { bg: '#3b82f620', color: '#3b82f6', label: 'Seatbelt' },
  triple_riding: { bg: '#ec489920', color: '#ec4899', label: 'Triple Riding' },
  distracted_driving: { bg: '#7c3aed20', color: '#7c3aed', label: 'Distracted' },
};

export default function ViolationBadge({ type }) {
  const config = TYPE_COLORS[type] || { bg: '#64748b20', color: '#64748b', label: type };
  return (
    <span className="badge" style={{ background: config.bg, color: config.color }}>
      {config.label}
    </span>
  );
}
