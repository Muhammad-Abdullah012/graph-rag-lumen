import React, { useState, useEffect } from 'react';
import './DocumentList.css';

function DocumentList() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/documents/list`);
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
    const interval = setInterval(fetchDocuments, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading && documents.length === 0) {
    return (
      <div className="document-list">
        <h2>Wissensgraph / Knowledge Graph</h2>
        <div className="loading-state">Laden...</div>
      </div>
    );
  }

  return (
    <div className="document-list">
      <div className="doc-list-header">
        <h2>Wissensgraph / Knowledge Graph</h2>
        <button className="btn-refresh" onClick={fetchDocuments}>
          ↻ Aktualisieren
        </button>
      </div>

      {error && <div className="error-message">⚠️ {error}</div>}

      {documents.length === 0 ? (
        <div className="empty-state">
          <p>Keine Dokumente vorhanden.</p>
          <p>Laden Sie ein PDF hoch, um den Wissensgraphen aufzubauen.</p>
        </div>
      ) : (
        <>
          <div className="doc-summary">
            <div className="summary-card">
              <span className="summary-number">{documents.length}</span>
              <span className="summary-label">Dokumente</span>
            </div>
            <div className="summary-card">
              <span className="summary-number">
                {documents.reduce((sum, d) => sum + (d.chunks_count || 0), 0)}
              </span>
              <span className="summary-label">Text-Chunks</span>
            </div>
            <div className="summary-card">
              <span className="summary-number">
                {documents.reduce((sum, d) => sum + (d.entities_count || 0), 0)}
              </span>
              <span className="summary-label">Entitäten</span>
            </div>
          </div>

          <div className="doc-table">
            <div className="doc-table-header">
              <span className="col-name">Dokument</span>
              <span className="col-chunks">Chunks</span>
              <span className="col-entities">Entitäten</span>
              <span className="col-status">Status</span>
            </div>
            {documents.map((doc, idx) => (
              <div key={idx} className="doc-table-row">
                <span className="col-name" title={doc.name}>
                  📄 {doc.name}
                </span>
                <span className="col-chunks">{doc.chunks_count}</span>
                <span className="col-entities">{doc.entities_count}</span>
                <span className={`col-status status-${doc.status}`}>
                  {doc.status === 'completed' ? '✅' : doc.status === 'processing' ? '⏳' : doc.status === 'failed' ? '❌' : '—'}
                  {' '}{doc.status}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default DocumentList;
