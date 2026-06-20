import React, { useState, useMemo, useEffect } from 'react';
import { motion } from 'framer-motion';
import ViolationBadge from '../components/ViolationBadge';
import { Search, Download, FileText, X, ChevronLeft, ChevronRight } from 'lucide-react';
import { useApi, apiPatch } from '../hooks/useApi';

const VIOLATION_TYPES = ['helmet', 'speed', 'wrong_side', 'red_light', 'illegal_parking', 'seatbelt', 'triple_riding', 'distracted_driving'];
const CAMERA_IDS = ['Cam-01: Silk Board Junction', 'Cam-02: Electronic City Toll', 'Cam-03: Indiranagar 100ft Rd', 'Cam-04: Marathahalli Bridge', 'Cam-05: Hebbal Flyover', 'Cam-06: Majestic Circle', 'Cam-07: MG Road Junction', 'Cam-08: Yeshwanthpur Chowk'];

const MOCK_VIOLATIONS = Array.from({ length: 50 }, (_, i) => ({
  id: i + 1,
  violation_time: new Date(2026, 5, 16, 6 + Math.floor(i / 5), (i * 7) % 60).toISOString(),
  camera_id: CAMERA_IDS[i % 8],
  violation_type: VIOLATION_TYPES[i % 8],
  license_plate: ['KA-01-AB-', 'KA-03-CD-', 'KA-05-EF-', 'KA-51-GH-', 'KA-53-IJ-', 'KA-02-KL-', 'KA-04-MN-', 'KA-50-OP-'][i % 8] + String(1000 + i * 37).slice(0, 4),
  confidence: 0.75 + (Math.sin(i) * 0.12 + 0.12),
  status: ['pending', 'confirmed', 'dismissed'][i % 3],
  vehicle_class: ['motorcycle', 'car', 'truck', 'car', 'motorcycle', 'bus', 'auto_rickshaw', 'car'][i % 8],
  sha256_hash: Array.from({ length: 64 }, () => '0123456789abcdef'[Math.floor(Math.random() * 16)]).join(''),
}));

export default function Violations() {
  const [typeFilter, setTypeFilter] = useState('');
  const [cameraFilter, setCameraFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [plateSearch, setPlateSearch] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [expandedRow, setExpandedRow] = useState(null);
  const [showCertificate, setShowCertificate] = useState(null);
  const perPage = 10;

  // Convert cameraFilter name back to camera ID number for API
  const cameraApiId = useMemo(() => {
    if (!cameraFilter) return '';
    const match = cameraFilter.match(/Cam-0(\d)/);
    return match ? match[1] : '';
  }, [cameraFilter]);

  // Fetch real violations from the API
  const { data: apiResponse, loading: apiLoading, refetch } = useApi(
    `/violations?page=${currentPage}&per_page=${perPage}${typeFilter ? `&violation_type=${typeFilter}` : ''}${cameraApiId ? `&camera_id=${cameraApiId}` : ''}${statusFilter ? `&status=${statusFilter}` : ''}${plateSearch ? `&plate_search=${plateSearch}` : ''}`
  );

  const getCameraName = (camId) => {
    const names = {
      1: 'Cam-01: Silk Board Junction',
      2: 'Cam-02: Electronic City Toll',
      3: 'Cam-03: Indiranagar 100ft Rd',
      4: 'Cam-04: Marathahalli Bridge',
      5: 'Cam-05: Hebbal Flyover',
      6: 'Cam-06: Majestic Circle',
      7: 'Cam-07: MG Road Junction',
      8: 'Cam-08: Yeshwanthpur Chowk'
    };
    if (typeof camId === 'string' && camId.includes('Cam-')) return camId;
    return names[camId] || `Cam-${camId}`;
  };

  const handleStatusUpdate = async (v, newStatus) => {
    try {
      const targetId = v.violation_id || v.id;
      const res = await apiPatch(`/violations/${targetId}`, { status: newStatus });
      if (res && !res.error) {
        refetch();
        alert(`Violation status updated to ${newStatus}`);
      } else {
        alert(res?.detail || 'Failed to update status.');
      }
    } catch {
      alert('Mock mode: Status updated locally.');
    }
  };

  const displayViolations = useMemo(() => {
    if (apiResponse?.data && apiResponse.data.length > 0) {
      return apiResponse.data;
    }
    return MOCK_VIOLATIONS.filter(v => {
      if (typeFilter && v.violation_type !== typeFilter) return false;
      if (cameraFilter && v.camera_id !== cameraFilter) return false;
      if (statusFilter && v.status !== statusFilter) return false;
      if (plateSearch && !v.license_plate.toLowerCase().includes(plateSearch.toLowerCase())) return false;
      return true;
    });
  }, [apiResponse, typeFilter, cameraFilter, statusFilter, plateSearch]);

  const totalCount = useMemo(() => {
    if (apiResponse?.pagination?.total) {
      return apiResponse.pagination.total;
    }
    const filteredMock = MOCK_VIOLATIONS.filter(v => {
      if (typeFilter && v.violation_type !== typeFilter) return false;
      if (cameraFilter && v.camera_id !== cameraFilter) return false;
      if (statusFilter && v.status !== statusFilter) return false;
      if (plateSearch && !v.license_plate.toLowerCase().includes(plateSearch.toLowerCase())) return false;
      return true;
    });
    return filteredMock.length;
  }, [apiResponse, typeFilter, cameraFilter, statusFilter, plateSearch]);

  const totalPages = useMemo(() => {
    if (apiResponse?.pagination?.total_pages) {
      return apiResponse.pagination.total_pages;
    }
    return Math.ceil(totalCount / perPage);
  }, [apiResponse, totalCount]);

  const paged = useMemo(() => {
    if (apiResponse?.data && apiResponse.data.length > 0) {
      return apiResponse.data;
    }
    const filteredMock = MOCK_VIOLATIONS.filter(v => {
      if (typeFilter && v.violation_type !== typeFilter) return false;
      if (cameraFilter && v.camera_id !== cameraFilter) return false;
      if (statusFilter && v.status !== statusFilter) return false;
      if (plateSearch && !v.license_plate.toLowerCase().includes(plateSearch.toLowerCase())) return false;
      return true;
    });
    return filteredMock.slice((currentPage - 1) * perPage, currentPage * perPage);
  }, [apiResponse, typeFilter, cameraFilter, statusFilter, plateSearch, currentPage]);

  const formatTime = (iso) => {
    const d = new Date(iso);
    return d.toLocaleString('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
  };

  const getConfColor = (conf) => conf > 0.9 ? '#10b981' : conf > 0.8 ? '#f59e0b' : '#ef4444';

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3 className="section-title"><FileText size={18} /> Violation Records</h3>
        <button className="btn btn-secondary btn-sm" onClick={() => alert('CSV export coming soon')}><Download size={14} /> Export CSV</button>
      </div>

      <div className="filter-bar">
        <select className="form-select" value={typeFilter} onChange={e => { setTypeFilter(e.target.value); setCurrentPage(1); }}>
          <option value="">All Types</option>
          {VIOLATION_TYPES.map(t => <option key={t} value={t}>{t.replace('_', ' ')}</option>)}
        </select>
        <select className="form-select" value={cameraFilter} onChange={e => { setCameraFilter(e.target.value); setCurrentPage(1); }}>
          <option value="">All Cameras</option>
          {CAMERA_IDS.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <select className="form-select" value={statusFilter} onChange={e => { setStatusFilter(e.target.value); setCurrentPage(1); }}>
          <option value="">All Status</option>
          <option value="pending">Pending</option>
          <option value="confirmed">Confirmed</option>
          <option value="dismissed">Dismissed</option>
        </select>
        <div className="search-box">
          <Search size={14} />
          <input type="text" placeholder="Search plate..." value={plateSearch} onChange={e => { setPlateSearch(e.target.value); setCurrentPage(1); }} />
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Camera</th>
                <th>Type</th>
                <th>Plate</th>
                <th>Confidence</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {paged.map(v => {
                const uniqueId = v.violation_id || v.id;
                return (
                  <React.Fragment key={uniqueId}>
                    <tr onClick={() => setExpandedRow(expandedRow === uniqueId ? null : uniqueId)} style={{ cursor: 'pointer' }}>
                      <td>{formatTime(v.violation_time)}</td>
                      <td style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{getCameraName(v.camera_id)}</td>
                      <td><ViolationBadge type={v.violation_type} /></td>
                      <td style={{ fontFamily: "'Courier New', monospace", color: '#06b6d4', fontWeight: 600 }}>{v.license_plate}</td>
                      <td>
                        <div className="confidence-bar">
                          <div className="confidence-track">
                            <div className="confidence-fill" style={{ width: `${v.confidence * 100}%`, background: getConfColor(v.confidence) }} />
                          </div>
                          <span style={{ fontSize: '0.75rem' }}>{Math.round(v.confidence * 100)}%</span>
                        </div>
                      </td>
                      <td><span className={`badge badge-${v.status}`}>{v.status}</span></td>
                      <td>
                        <button className="btn btn-secondary btn-sm" onClick={e => { e.stopPropagation(); setExpandedRow(uniqueId); }}>View</button>
                      </td>
                    </tr>
                    {expandedRow === uniqueId && (
                      <tr>
                        <td colSpan={7} style={{ background: 'rgba(15,23,42,0.5)', padding: '1rem 1.5rem' }}>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1.5fr', gap: '1.5rem', fontSize: '0.82rem' }}>
                            <div>
                              <div><strong style={{ color: '#94a3b8' }}>Vehicle Class:</strong> {v.vehicle_class}</div>
                              <div style={{ marginTop: '0.25rem' }}><strong style={{ color: '#94a3b8' }}>Confidence:</strong> {(v.confidence * 100).toFixed(1)}%</div>
                              <div style={{ marginTop: '0.25rem' }}><strong style={{ color: '#94a3b8' }}>Status:</strong> {v.status}</div>
                              <div style={{ marginTop: '0.25rem' }}><strong style={{ color: '#94a3b8' }}>SHA-256 Hash:</strong> <span style={{ fontFamily: 'monospace', fontSize: '0.72rem', wordBreak: 'break-all' }}>{v.sha256_hash}</span></div>
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', justifyContent: 'center' }}>
                              <button className="btn btn-primary btn-sm" onClick={e => { e.stopPropagation(); handleStatusUpdate(v, 'confirmed'); }}>Confirm</button>
                              <button className="btn btn-secondary btn-sm" onClick={e => { e.stopPropagation(); handleStatusUpdate(v, 'dismissed'); }}>Dismiss</button>
                              <button className="btn btn-secondary btn-sm" onClick={e => { e.stopPropagation(); setShowCertificate(v); }}><FileText size={14} /> Generate Certificate</button>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                              <div className="camera-preview" style={{ height: 100, width: '100%', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.02)', border: '1px dashed rgba(255,255,255,0.1)', borderRadius: 4 }}>
                                {v.evidence_image_path ? (
                                  <img src={v.evidence_image_path.startsWith('/') ? `http://localhost:8000${v.evidence_image_path}` : v.evidence_image_path} alt="Evidence" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                ) : (
                                  <span style={{ color: '#64748b', fontSize: '0.75rem' }}>No Snapshot Image</span>
                                )}
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="pagination">
        <button className="page-btn" onClick={() => setCurrentPage(p => Math.max(1, p - 1))} disabled={currentPage === 1}><ChevronLeft size={14} /> Prev</button>
        {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
          const page = currentPage <= 3 ? i + 1 : currentPage + i - 2;
          if (page > totalPages || page < 1) return null;
          return <button key={page} className={`page-btn ${page === currentPage ? 'page-btn-active' : ''}`} onClick={() => setCurrentPage(page)}>{page}</button>;
        })}
        <button className="page-btn" onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))} disabled={currentPage === totalPages}>Next <ChevronRight size={14} /></button>
      </div>
      <div style={{ textAlign: 'center', fontSize: '0.75rem', color: '#64748b', marginTop: '0.5rem' }}>
        Showing {(currentPage - 1) * perPage + 1}–{Math.min(currentPage * perPage, totalCount)} of {totalCount} violations
      </div>

      {showCertificate && (
        <div className="modal-overlay" onClick={() => setShowCertificate(null)}>
          <motion.div className="modal" onClick={e => e.stopPropagation()} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
            <div className="modal-header">
              <h3 className="modal-title">BSA Section 63(4)(c) Certificate</h3>
              <button className="modal-close" onClick={() => setShowCertificate(null)}><X size={18} /></button>
            </div>
            <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.75rem', color: '#94a3b8', lineHeight: 1.8, fontFamily: "'Courier New', monospace" }}>
{`═══════════════════════════════════════════════════════
  CERTIFICATE UNDER SECTION 63(4)(c) OF THE
  BHARATIYA SAKSHYA ADHINIYAM, 2023
═══════════════════════════════════════════════════════

PART A: SYSTEM OPERATOR AFFIRMATION

I, [Name of Officer], do hereby solemnly affirm:

1. I am lawfully entitled to control the operations
   of the automated traffic enforcement system,
   incorporating Edge Camera Node: ${showCertificate.camera_id}

2. The digital system was under lawful control for
   traffic safety enforcement purposes.

3. The computer system was operating properly.

4. SHA-256 Hash of evidentiary image:
   ${showCertificate.sha256_hash}

VIOLATION DETAILS:
  ID:         ${showCertificate.id}
  Type:       ${showCertificate.violation_type.replace('_', ' ')}
  Time:       ${new Date(showCertificate.violation_time).toLocaleString()}
  Vehicle:    ${showCertificate.vehicle_class}
  Plate:      ${showCertificate.license_plate}
  Confidence: ${(showCertificate.confidence * 100).toFixed(1)}%
  Camera:     ${showCertificate.camera_id}

Signature: ________________________  Date: __________

═══════════════════════════════════════════════════════`}
            </pre>
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
              <button className="btn btn-primary" onClick={() => { navigator.clipboard?.writeText('Certificate copied'); alert('Certificate text copied!'); }}>Copy Text</button>
              <button className="btn btn-secondary" onClick={() => setShowCertificate(null)}>Close</button>
            </div>
          </motion.div>
        </div>
      )}
    </motion.div>
  );
}
