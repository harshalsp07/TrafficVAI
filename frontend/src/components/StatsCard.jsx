import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

export default function StatsCard({ icon: Icon, value, label, trend, trendValue, color = '#3b82f6' }) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const numVal = typeof value === 'number' ? value : parseFloat(value) || 0;
    const duration = 1000;
    const steps = 30;
    const increment = numVal / steps;
    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if (current >= numVal) {
        setDisplayValue(numVal);
        clearInterval(timer);
      } else {
        setDisplayValue(Math.floor(current));
      }
    }, duration / steps);
    return () => clearInterval(timer);
  }, [value]);

  return (
    <motion.div
      className="card stats-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.3 }}
    >
      <div className="stats-card-header">
        <div className="stats-icon" style={{ background: `${color}20`, color }}>
          {Icon && <Icon size={22} />}
        </div>
        {trend && (
          <span className={`stats-trend ${trend === 'up' ? 'trend-up' : 'trend-down'}`}>
            {trend === 'up' ? '↑' : '↓'} {trendValue}
          </span>
        )}
      </div>
      <div className="stats-value" style={{ color }}>
        {typeof value === 'string' ? value : displayValue.toLocaleString()}
      </div>
      <div className="stats-label">{label}</div>
    </motion.div>
  );
}
