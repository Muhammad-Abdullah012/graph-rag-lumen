import React from 'react';
import QAChat from './components/QAChat';
import HealthStatus from './components/HealthStatus';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Lumen IT POC Eurocode Betonbau</h1>
        <p>Knowledge Base Question & Answering</p>
        <HealthStatus />
      </header>

      <main className="app-content">
        <QAChat />
      </main>
    </div>
  );
}

export default App;
