import React, { useState, useEffect, useCallback } from 'react';
import './ConversationSidebar.css';

function ConversationSidebar({ activeId, onSelect, onNew, refreshSignal }) {
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(true);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const fetchConversations = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/conversations/`);
      if (response.ok) {
        const data = await response.json();
        setConversations(data);
      }
    } catch (err) {
      console.error('Failed to load conversations:', err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL]);

  useEffect(() => {
    fetchConversations();
  }, [fetchConversations, refreshSignal]);

  const handleNew = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/conversations/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Conversation' }),
      });
      if (response.ok) {
        const data = await response.json();
        await fetchConversations();
        onSelect(data.id);
        if (onNew) onNew(data.id);
      }
    } catch (err) {
      console.error('Failed to create conversation:', err);
    }
  };

  const handleDelete = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm('Delete this conversation?')) return;
    try {
      await fetch(`${API_BASE_URL}/api/conversations/${id}`, {
        method: 'DELETE',
      });
      await fetchConversations();
      if (activeId === id) onSelect(null);
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  };

  const formatDate = (isoStr) => {
    const d = new Date(isoStr);
    const now = new Date();
    const diffMs = now - d;
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    if (diffDay < 7) return `${diffDay}d ago`;
    return d.toLocaleDateString();
  };

  return (
    <div className="conversation-sidebar">
      <div className="sidebar-header">
        <h3>Conversations</h3>
        <button className="btn-new-chat" onClick={handleNew} title="New conversation">
          +
        </button>
      </div>

      <div className="conversation-list">
        {loading ? (
          <div className="sidebar-loading">Loading...</div>
        ) : conversations.length === 0 ? (
          <div className="sidebar-empty">
            <p>No conversations yet.</p>
            <p>Click + to start one.</p>
          </div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${conv.id === activeId ? 'active' : ''}`}
              onClick={() => onSelect(conv.id)}
            >
              <div className="conv-title">{conv.title}</div>
              <div className="conv-meta">
                <span className="conv-time">{formatDate(conv.updated_at)}</span>
                <button
                  className="btn-delete-conv"
                  onClick={(e) => handleDelete(e, conv.id)}
                  title="Delete"
                >
                  ✕
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default ConversationSidebar;
