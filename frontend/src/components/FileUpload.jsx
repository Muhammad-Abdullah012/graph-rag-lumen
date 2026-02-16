import React, { useState, useEffect, useCallback } from 'react';
import './FileUpload.css';

function FileUpload() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const fetchFiles = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/files/`);
      if (response.ok) {
        const data = await response.json();
        setFiles(data);
      }
    } catch (err) {
      console.error('Failed to load files:', err);
    }
  }, [API_BASE_URL]);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const handleUpload = async (fileList) => {
    if (!fileList || fileList.length === 0) return;
    setUploading(true);
    setStatusMsg('Uploading...');

    try {
      const formData = new FormData();
      for (const f of fileList) {
        formData.append('files', f);
      }

      const response = await fetch(`${API_BASE_URL}/api/files/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await response.json();
      setStatusMsg(data.message);
      await fetchFiles();
    } catch (err) {
      setStatusMsg(`Error: ${err.message}`);
    } finally {
      setUploading(false);
      setTimeout(() => setStatusMsg(''), 4000);
    }
  };

  const handleFileInput = (e) => {
    handleUpload(e.target.files);
    e.target.value = '';
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleUpload(e.dataTransfer.files);
  };

  const handleDelete = async (id) => {
    try {
      await fetch(`${API_BASE_URL}/api/files/${id}`, { method: 'DELETE' });
      await fetchFiles();
    } catch (err) {
      console.error('Failed to delete file:', err);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="file-upload-panel">
      <h3>Files</h3>

      <div
        className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <span>{uploading ? 'Uploading...' : 'Drop files here or click to upload'}</span>
        <input
          type="file"
          multiple
          onChange={handleFileInput}
          disabled={uploading}
          className="drop-input"
        />
      </div>

      {statusMsg && <div className="upload-status">{statusMsg}</div>}

      {files.length > 0 && (
        <div className="file-list">
          {files.map((f) => (
            <div key={f.id} className="file-item">
              <div className="file-info">
                <span className="file-name" title={f.original_filename}>
                  {f.original_filename}
                </span>
                <span className="file-size">{formatSize(f.file_size)}</span>
              </div>
              <button
                className="btn-delete-file"
                onClick={() => handleDelete(f.id)}
                title="Delete file"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default FileUpload;
