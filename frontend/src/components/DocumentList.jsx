import React, { useEffect, useState } from 'react';
import './DocumentList.css';

function DocumentList({ refreshKey = 0 }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const buildFileUrl = (path) => {
    if (!path) return '#';
    return path.startsWith('http') ? path : `${API_BASE_URL}${path.startsWith('/') ? '' : '/'}${path}`;
  };

  const formatSize = (bytes) => (typeof bytes === 'number' ? `${(bytes / (1024 * 1024)).toFixed(2)} MB` : '—');

  const formatDate = (value) => (value ? new Date(value).toLocaleString() : '—');

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/documents`);
      if (!response.ok) throw new Error('Failed to fetch documents');
      const data = await response.json();
      setDocuments(data.documents || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [refreshKey]);

  if (loading && documents.length === 0) {
    return (
      <div className="document-list">
        <h2>Uploaded PDFs</h2>
        <div className="loading-state">Laden...</div>
      </div>
    );
  }

  return (
    <div className="document-list">
      <div className="doc-list-header">
        <h2>Uploaded PDFs</h2>
        <button className="btn-refresh" onClick={fetchDocuments}>
          ↻ Refresh
        </button>
      </div>

      {error && <div className="error-message">⚠️ {error}</div>}

      {documents.length === 0 ? (
        <div className="empty-state">
          <p>No PDFs uploaded yet.</p>
          <p>Upload a PDF to store it on the server.</p>
        </div>
      ) : (
        <div className="doc-table">
          <div className="doc-table-header">
            <span className="col-name">Document</span>
            <span className="col-size">Size</span>
            <span className="col-date">Uploaded</span>
            <span className="col-actions">Actions</span>
          </div>
          {documents.map((doc) => (
            <div key={doc.stored_filename} className="doc-table-row">
              <span className="col-name" title={doc.filename}>
                📄 {doc.filename}
              </span>
              <span className="col-size">{formatSize(doc.size_bytes)}</span>
              <span className="col-date">{formatDate(doc.uploaded_at)}</span>
              <span className="col-actions">
                <a href={buildFileUrl(doc.url)} target="_blank" rel="noreferrer">
                  Download
                </a>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default DocumentList;
