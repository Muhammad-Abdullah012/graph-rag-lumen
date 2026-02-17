import React, { useState, useCallback, useEffect } from 'react';
import QAChat from './components/QAChat';
import HealthStatus from './components/HealthStatus';
import ConversationSidebar from './components/ConversationSidebar';
import DocumentList from './components/DocumentList';
import DocumentUpload from './components/DocumentUpload';

function App() {
  const [activeConversation, setActiveConversation] = useState(null);
  const [refreshSignal, setRefreshSignal] = useState(0);
  const [activeTab, setActiveTab] = useState('chat');
  const [docRefreshKey, setDocRefreshKey] = useState(0);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Restore last-used tab from local storage (keep) and conversation from URL query
  useEffect(() => {
    const storedTab = localStorage.getItem('lumen-active-tab');
    if (storedTab) setActiveTab(storedTab);

    const params = new URLSearchParams(window.location.search);
    const conv = params.get('conversation');
    if (conv) setActiveConversation(conv);
  }, []);

  const updateConversationQuery = useCallback((id) => {
    const url = new URL(window.location.href);
    if (id) {
      url.searchParams.set('conversation', id);
    } else {
      url.searchParams.delete('conversation');
    }
    window.history.replaceState({}, '', url.toString());
    setActiveConversation(id);
  }, []);

  const handleTabChange = useCallback((tab) => {
    setActiveTab(tab);
    localStorage.setItem('lumen-active-tab', tab);
  }, []);

  // Called by QAChat when a new conversation is created on first message
  const handleConversationCreated = useCallback((newId) => {
    updateConversationQuery(newId);
    setRefreshSignal((s) => s + 1);
  }, [updateConversationQuery]);

  // Called by QAChat after first message to set the title
  const handleFirstMessage = useCallback(
    async (text, convId) => {
      const id = convId || activeConversation;
      if (!id) return;
      const title = text.length > 50 ? text.slice(0, 50) + '...' : text;
      try {
        await fetch(`${API_BASE_URL}/api/conversations/${id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title }),
        });
        setRefreshSignal((s) => s + 1);
      } catch {
        // ignore
      }
    },
    [activeConversation, API_BASE_URL]
  );

  // Start a fresh chat (clear selection = draft mode)
  const handleNewChat = useCallback(() => {
    updateConversationQuery(null);
  }, [updateConversationQuery]);

  const handleConversationSelect = useCallback(
    (id) => {
      updateConversationQuery(id);
    },
    [updateConversationQuery]
  );

  const handleUploadComplete = useCallback(() => {
    setDocRefreshKey((k) => k + 1);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Lumen IT POC Eurocode Betonbau</h1>
        <p>Knowledge Base Question & Answering</p>
        <HealthStatus />
      </header>

      <div className="tab-bar">
        <button
          className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => handleTabChange('chat')}
        >
          Chat
        </button>
        <button
          className={`tab-btn ${activeTab === 'files' ? 'active' : ''}`}
          onClick={() => handleTabChange('files')}
        >
          Files
        </button>
        <button
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => handleTabChange('upload')}
        >
          Upload
        </button>
      </div>

      {activeTab === 'chat' && (
        <div className="app-body">
          <div className="sidebar-col">
            <ConversationSidebar
              activeId={activeConversation}
              onSelect={handleConversationSelect}
              onNewChat={handleNewChat}
              refreshSignal={refreshSignal}
            />
          </div>

          <main className="app-content">
            <QAChat
              conversationId={activeConversation}
              onFirstMessage={handleFirstMessage}
              onConversationCreated={handleConversationCreated}
            />
          </main>
        </div>
      )}

      {activeTab === 'files' && (
        <div className="tab-panel">
          <DocumentList refreshKey={docRefreshKey} />
        </div>
      )}

      {activeTab === 'upload' && (
        <div className="tab-panel">
          <DocumentUpload onUpload={handleUploadComplete} />
        </div>
      )}
    </div>
  );
}

export default App;
