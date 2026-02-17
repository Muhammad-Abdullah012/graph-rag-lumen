import React, { useState } from 'react';
import './DocumentUpload.css';

function DocumentUpload({ onUploadComplete }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState('');
  const [uploadedDoc, setUploadedDoc] = useState(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
      setStatus('Please select a PDF file');
      return;
    }

    if (selectedFile.size > 50 * 1024 * 1024) {
      setStatus('File too large (max 50MB)');
      return;
    }

    setFile(selectedFile);
    setStatus('');
    setUploadedDoc(null);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setStatus('Please select a file first');
      return;
    }

    setUploading(true);
    setStatus('Uploading...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let detail = 'Upload failed';
        try {
          const errorBody = await response.json();
          detail = errorBody.detail || detail;
        } catch (err) {
          // Ignore JSON parse errors, fall back to default message
        }
        throw new Error(detail);
      }

      const data = await response.json();
      setStatus('Upload complete');
      setUploadedDoc(data);
      setFile(null);

      if (typeof onUploadComplete === 'function') {
        onUploadComplete();
      }
    } catch (error) {
      setStatus(error.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="document-upload">
      <h2>Upload PDF</h2>
      <p className="description">Store your PDF so it can be used later.</p>

      <form onSubmit={handleUpload} className="upload-form">
        <div className="file-input-wrapper">
          <input
            type="file"
            id="file-input"
            accept=".pdf"
            onChange={handleFileSelect}
            disabled={uploading}
          />
          <label htmlFor="file-input" className="file-label">
            {file ? file.name : 'Choose a PDF file'}
          </label>
        </div>

        <button type="submit" className="btn-upload" disabled={!file || uploading}>
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
      </form>

      {status && <div className="status-message">{status}</div>}

      {uploadedDoc && (
        <div className="upload-info">
          <h3>Stored file</h3>
          <ul>
            <li>Name: {uploadedDoc.filename}</li>
            <li>Size: {(uploadedDoc.size_bytes / (1024 * 1024)).toFixed(2)} MB</li>
            <li>
              Link:{' '}
              <a href={`${API_BASE_URL}${uploadedDoc.url}`} target="_blank" rel="noreferrer">
                {uploadedDoc.url}
              </a>
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}

export default DocumentUpload;
