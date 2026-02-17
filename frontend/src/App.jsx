import React, { useState } from 'react';
import DocumentList from './components/DocumentList';
import DocumentUpload from './components/DocumentUpload';
import HealthStatus from './components/HealthStatus';
import QAChat from './components/QAChat';

function App() {
  const [activeTab, setActiveTab] = useState('files');
  const [refreshKey, setRefreshKey] = useState(0);

  const handleUploadComplete = () => setRefreshKey((prev) => prev + 1);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Lumen IT POC Eurocode Betonbau</h1>
        <p>Knowledge Base Question & Answering</p>
        <HealthStatus />
      </header>

      <div className="app-container">
        <nav className="nav-tabs">
          <button
            className={`nav-tab ${activeTab === 'files' ? 'active' : ''}`}
            onClick={() => setActiveTab('files')}
          >
            Dateien
          </button>
          <button
            className={`nav-tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            Chat
          </button>
        </nav>

        <main className="app-content">
          {activeTab === 'files' ? (
            <div className="files-layout">
              <DocumentUpload onUploadComplete={handleUploadComplete} />
              <DocumentList refreshKey={refreshKey} />
            </div>
          ) : (
            <QAChat />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
