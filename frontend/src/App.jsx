import React, { useState, useCallback } from 'react';
import QAChat from './components/QAChat';
import HealthStatus from './components/HealthStatus';
import ConversationSidebar from './components/ConversationSidebar';
import FileUpload from './components/FileUpload';

function App() {
  const [activeConversation, setActiveConversation] = useState(null);
  const [refreshSignal, setRefreshSignal] = useState(0);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handleFirstMessage = useCallback(
    async (text) => {
      // Auto-set conversation title from first message
      if (!activeConversation) return;
      const title = text.length > 50 ? text.slice(0, 50) + '...' : text;
      try {
        await fetch(`${API_BASE_URL}/api/conversations/${activeConversation}`, {
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

  return (
    <div className="app">
      <header className="app-header">
        <h1>Lumen IT POC Eurocode Betonbau</h1>
        <p>Knowledge Base Question & Answering</p>
        <HealthStatus />
      </header>

      <div className="app-body">
        <div className="sidebar-col">
          <ConversationSidebar
            activeId={activeConversation}
            onSelect={setActiveConversation}
            refreshSignal={refreshSignal}
          />
          <FileUpload />
        </div>

        <main className="app-content">
          <QAChat
            conversationId={activeConversation}
            onFirstMessage={handleFirstMessage}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
