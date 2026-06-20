import React, { useState, useEffect } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { LayoutDashboard, AlertTriangle, BarChart3, Camera, Brain, Settings, Bell, Search, Menu, X } from 'lucide-react';
import StatusDot from './StatusDot';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/violations', label: 'Violations', icon: AlertTriangle },
  { path: '/analytics', label: 'Analytics', icon: BarChart3 },
  { path: '/cameras', label: 'Cameras', icon: Camera },
  { path: '/training', label: 'Training', icon: Brain },
  { path: '/settings', label: 'Settings', icon: Settings },
];

const PAGE_TITLES = {
  '/': 'Dashboard',
  '/violations': 'Violations',
  '/analytics': 'Analytics',
  '/cameras': 'Cameras',
  '/training': 'Training',
  '/settings': 'Settings',
};

export default function Layout() {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [time, setTime] = useState(new Date());
  const [notificationCount] = useState(3);

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const pageTitle = PAGE_TITLES[location.pathname] || 'TrafficAI';

  return (
    <div className="layout">
      <aside className={`sidebar ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
        <div className="sidebar-header">
          <h1 className="sidebar-logo">TrafficAI</h1>
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
        <nav className="sidebar-nav">
          {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              end={path === '/'}
              className={({ isActive }) => `nav-link ${isActive ? 'nav-link-active' : ''}`}
            >
              <Icon size={20} />
              {sidebarOpen && <span>{label}</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <StatusDot status="online" />
          {sidebarOpen && <span className="sidebar-status-text">System Online</span>}
        </div>
      </aside>
      <div className="main-wrapper">
        <header className="top-header">
          <h2 className="page-title">{pageTitle}</h2>
          <div className="header-actions">
            <div className="search-box">
              <Search size={16} />
              <input type="text" placeholder="Search violations, plates..." />
            </div>
            <button className="notification-btn">
              <Bell size={20} />
              {notificationCount > 0 && <span className="notification-badge">{notificationCount}</span>}
            </button>
            <span className="header-time">
              {time.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
          </div>
        </header>
        <main className="main-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
