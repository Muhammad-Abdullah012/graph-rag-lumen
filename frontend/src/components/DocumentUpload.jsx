import React, { useState } from 'react';
import './DocumentUpload.css';

function DocumentUpload({ onUpload }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState('');
  const [uploadedDocs, setUploadedDocs] = useState([]);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
        setStatus('❌ Please select a PDF file');
        return;
      }
      if (selectedFile.size > 50 * 1024 * 1024) {
        setStatus('❌ File too large (max 50MB)');
        return;
      }
      setFile(selectedFile);
      setStatus('');
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setStatus('❌ Please select a file first');
      return;
    }

    setUploading(true);
    setStatus('⏳ Uploading...');

    try {
      const formData = new FormData();
      formData.append('files', file);

      const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }

      const data = await response.json();
      setStatus(data.message || '✅ Uploaded');
      setUploadedDocs(data.documents || []);
      onUpload?.(data);
      setFile(null);
    } catch (error) {
      setStatus(`❌ Upload failed: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="document-upload">
      <h2>Upload PDF Document</h2>
      <p className="description">
        Store PDFs for future processing. Files are kept in the documents area.
      </p>

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
            {file ? `${file.name}` : 'Choose PDF file'}
          </label>
        </div>

        <button type="submit" className="btn-upload" disabled={!file || uploading}>
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
      </form>

      {status && <div className="status-message">{status}</div>}

      {uploadedDocs.length > 0 && (
        <div className="processing-details">
          <h3>Uploaded</h3>
          <div className="status-info">
            {uploadedDocs.map((d) => (
              <p key={d.id}>
                <strong>{d.original_filename}</strong> ({(d.file_size / (1024 * 1024)).toFixed(2)} MB)
              </p>
            ))}
          </div>
        </div>
      )}

      <div className="upload-info">
        <h3>Notes</h3>
        <ul>
          <li>Files are stored only; processing will be added later.</li>
          <li>Maximum size {50} MB per file.</li>
        </ul>
      </div>
    </div>
  );
}

export default DocumentUpload;
