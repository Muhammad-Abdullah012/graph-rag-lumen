import React, { useState } from 'react';
import './DocumentUpload.css';

function DocumentUpload({ onUpload }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState('');
  const [documentId, setDocumentId] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);

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
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }

      const data = await response.json();
      setDocumentId(data.document_id);
      setStatus('✅ Uploaded! Processing...');

      // Poll for processing status
      pollProcessingStatus(data.document_id);

      onUpload(data);
      setFile(null);
      setUploading(false);
    } catch (error) {
      setStatus(`❌ Upload failed: ${error.message}`);
      setUploading(false);
    }
  };

  const pollProcessingStatus = (docId) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/documents/status/${docId}`);
        const data = await response.json();
        
        setProcessingStatus(data);

        if (data.status === 'completed') {
          setStatus('✅ Processing completed!');
          clearInterval(interval);
        } else if (data.status === 'failed') {
          setStatus(`❌ Processing failed: ${data.message}`);
          clearInterval(interval);
        } else {
          setStatus(`⏳ ${data.message}`);
        }
      } catch (error) {
        console.error('Error checking status:', error);
      }
    }, 2000);
  };

  return (
    <div className="document-upload">
      <h2>Upload PDF Document</h2>
      <p className="description">
        Upload a PDF document to extract text and build a knowledge graph
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
            {file ? `📄 ${file.name}` : '📁 Choose PDF file'}
          </label>
        </div>

        <button type="submit" className="btn-upload" disabled={!file || uploading}>
          {uploading ? '⏳ Uploading...' : '📤 Upload & Process'}
        </button>
      </form>

      {status && <div className="status-message">{status}</div>}

      {processingStatus && (
        <div className="processing-details">
          <h3>Processing Status</h3>
          <div className="status-info">
            <p><strong>Status:</strong> {processingStatus.status}</p>
            <p><strong>Message:</strong> {processingStatus.message}</p>
          </div>
        </div>
      )}

      <div className="upload-info">
        <h3>Supported Features:</h3>
        <ul>
          <li>✓ Text extraction with OCR</li>
          <li>✓ Table structure preservation</li>
          <li>✓ Code enrichment</li>
          <li>✓ Image classification</li>
          <li>✓ Knowledge graph creation</li>
          <li>✓ Vector embeddings</li>
        </ul>
      </div>
    </div>
  );
}

export default DocumentUpload;
