import React, { useState, useEffect, useCallback } from 'react';
import './DocumentList.css';

function DocumentList({ refreshKey = 0 }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/documents`);
      if (!response.ok) throw new Error('Failed to fetch documents');
      const data = await response.json();
      setDocuments(Array.isArray(data) ? data : data.documents || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL]);

  useEffect(() => {
    fetchDocuments();
    const interval = setInterval(fetchDocuments, 15000);
    return () => clearInterval(interval);
  }, [fetchDocuments, refreshKey]);

  return (
    <div className="document-list">
      <div className="doc-list-header">
        <h2>Documents</h2>
        <button className="btn-refresh" onClick={fetchDocuments}>
          ↻ Refresh
        </button>
      </div>

      {error && <div className="error-message">⚠️ {error}</div>}

      {loading && documents.length === 0 && (
        <div className="loading-state">Loading...</div>
      )}

      {documents.length === 0 ? (
        <div className="empty-state">
          <p>No documents uploaded yet.</p>
          <p>Upload PDFs to make them available for processing later.</p>
        </div>
      ) : (
        <>
          <div className="doc-summary">
            <div className="summary-card">
              <span className="summary-number">{documents.length}</span>
              <span className="summary-label">Documents</span>
            </div>
            <div className="summary-card">
              <span className="summary-number">
                {(documents.reduce((sum, d) => sum + (d.file_size || 0), 0) / (1024 * 1024)).toFixed(2)}
              </span>
              <span className="summary-label">MB total</span>
            </div>
          </div>

          <div className="doc-table">
            <div className="doc-table-header">
              <span className="col-name">Document</span>
              <span className="col-size">Size</span>
              <span className="col-type">Type</span>
              <span className="col-uploaded">Uploaded</span>
            </div>
            <div className="doc-table-body">
              {documents.map((doc) => (
                <div key={doc.id} className="doc-table-row">
                  <span className="col-name" title={doc.original_filename}>
                    📄 {doc.original_filename}
                  </span>
                  <span className="col-size">{(doc.file_size / (1024 * 1024)).toFixed(2)} MB</span>
                  <span className="col-type">{doc.content_type || '—'}</span>
                  <span className="col-uploaded">{new Date(doc.uploaded_at).toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default DocumentList;
