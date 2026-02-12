import React, { useState, useEffect } from 'react';
// import './App.css';
import DocumentUpload from './components/DocumentUpload';
import QAChat from './components/QAChat';
import HealthStatus from './components/HealthStatus';

function App() {
  const [documents, setDocuments] = useState([]);
  const [activeTab, setActiveTab] = useState('qa');

  const handleDocumentUpload = (newDoc) => {
    setDocuments([...documents, newDoc]);
  };

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
            className={`nav-tab ${activeTab === 'qa' ? 'active' : ''}`}
            onClick={() => setActiveTab('qa')}
          >
            💬 Ask Questions
          </button>
          <button 
            className={`nav-tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            📄 Upload Documents
          </button>
        </nav>

        <main className="app-content">
          {activeTab === 'qa' && <QAChat />}
          {activeTab === 'upload' && (
            <DocumentUpload onUpload={handleDocumentUpload} />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
