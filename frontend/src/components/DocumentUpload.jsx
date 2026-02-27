import React, { useState } from 'react';
import './DocumentUpload.css';

function DocumentUpload({ onUploadComplete }) {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [fileStatuses, setFileStatuses] = useState({}); // filename → status string

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handleFileSelect = (e) => {
    const selected = Array.from(e.target.files || []);
    const valid = [];
    const errors = {};

    for (const f of selected) {
      if (!f.name.toLowerCase().endsWith('.pdf')) {
        errors[f.name] = 'Not a PDF';
      } else if (f.size > 50 * 1024 * 1024) {
        errors[f.name] = 'Too large (max 50 MB)';
      } else {
        valid.push(f);
      }
    }

    setFiles(valid);
    setFileStatuses(errors);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (files.length === 0) return;

    setUploading(true);

    // Mark all valid files as pending
    const pending = {};
    files.forEach(f => { pending[f.name] = 'Uploading…'; });
    setFileStatuses(pending);

    try {
      const formData = new FormData();
      files.forEach(f => formData.append('files', f));

      const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let detail = 'Upload failed';
        try { detail = (await response.json()).detail || detail; } catch (_) {}
        throw new Error(detail);
      }

      const results = await response.json(); // List[DocumentUploadResponse]
      const statuses = {};
      results.forEach(doc => {
        statuses[doc.filename] = doc.message || 'Uploaded';
      });
      setFileStatuses(statuses);
      setFiles([]);

      if (typeof onUploadComplete === 'function') {
        onUploadComplete();
      }
    } catch (error) {
      const errStatuses = {};
      files.forEach(f => { errStatuses[f.name] = error.message || 'Upload failed'; });
      setFileStatuses(errStatuses);
    } finally {
      setUploading(false);
    }
  };

  const hasFiles = files.length > 0;
  const statusEntries = Object.entries(fileStatuses);

  return (
    <div className="document-upload">
      <h2>Upload PDF</h2>
      <p className="description">Select one or more PDF files to process and add to the knowledge graph.</p>

      <form onSubmit={handleUpload} className="upload-form">
        <div className="file-input-wrapper">
          <input
            type="file"
            id="file-input"
            accept=".pdf"
            multiple
            onChange={handleFileSelect}
            disabled={uploading}
          />
          <label htmlFor="file-input" className="file-label">
            {hasFiles
              ? `${files.length} file${files.length > 1 ? 's' : ''} selected`
              : 'Choose PDF file(s)'}
          </label>
        </div>

        <button type="submit" className="btn-upload" disabled={!hasFiles || uploading}>
          {uploading ? 'Uploading…' : `Upload${files.length > 1 ? ` (${files.length})` : ''}`}
        </button>
      </form>

      {statusEntries.length > 0 && (
        <ul className="file-status-list">
          {statusEntries.map(([name, msg]) => (
            <li key={name}>
              <span className="file-status-name">{name}</span>
              <span className="file-status-msg">{msg}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default DocumentUpload;
