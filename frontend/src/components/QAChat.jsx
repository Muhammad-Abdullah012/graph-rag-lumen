import React, { useState, useRef, useEffect, useCallback } from 'react';
import './QAChat.css';

function QAChat({ conversationId, onFirstMessage, onConversationCreated }) {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Create a new conversation via API
  const createConversation = async () => {
    const response = await fetch(`${API_BASE_URL}/api/conversations/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'New Conversation' }),
    });
    if (!response.ok) throw new Error('Failed to create conversation');
    const data = await response.json();
    return data.id;
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load history when conversationId changes
  const loadHistory = useCallback(async () => {
    if (!conversationId) {
      setMessages([]);
      return;
    }
    setLoadingHistory(true);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/conversations/${conversationId}/messages`
      );
      if (response.ok) {
        const data = await response.json();
        const mapped = data
          .filter((m) => m.role === 'human' || m.role === 'ai')
          .map((m) => ({
            role: m.role === 'human' ? 'user' : 'assistant',
            content: m.content,
            tool_calls: m.tool_calls || [],
          }));
        setMessages(mapped);
      }
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoadingHistory(false);
    }
  }, [conversationId, API_BASE_URL]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // Focus input when conversation changes
  useEffect(() => {
    inputRef.current?.focus();
  }, [conversationId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const userMessage = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMessage]);
    const currentQuestion = question;
    setQuestion('');
    setLoading(true);

    // If no conversation yet, create one first (ChatGPT-like auto-create)
    let activeId = conversationId;
    if (!activeId) {
      try {
        activeId = await createConversation();
        if (onConversationCreated) onConversationCreated(activeId);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Error: ${err.message}` },
        ]);
        setLoading(false);
        return;
      }
    }

    // Notify parent about first message (for auto-title)
    if (messages.length === 0 && onFirstMessage) {
      onFirstMessage(currentQuestion, activeId);
    }

    try {
      // Use non-streaming endpoint
      const response = await fetch(
        `${API_BASE_URL}/api/conversations/${activeId}/chat`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: currentQuestion }),
        }
      );

      if (!response.ok) throw new Error('Failed to get answer');

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || '',
          tools_used: data.tools_used || [],
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${error.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="qa-chat">
      <div className="chat-messages">
        {loadingHistory ? (
          <div className="empty-chat">
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
            <p>Loading conversation...</p>
          </div>
        ) : messages.length === 0 ? (
          <div className="empty-chat">
            <h2>Frage stellen / Ask a Question</h2>
            <p>Stellen Sie Fragen zum Eurocode-Wissensgraphen</p>
            <div className="example-questions">
              <h3>Beispielfragen:</h3>
              <ul>
                <li onClick={() => setQuestion('Was bedeutet das Symbol γf?')}>
                  "Was bedeutet das Symbol γf?"
                </li>
                <li onClick={() => setQuestion('Wie lautet die Formel für AEd?')}>
                  "Wie lautet die Formel für AEd?"
                </li>
                <li onClick={() => setQuestion('Was bedeutet die Abkürzung EQU?')}>
                  "Was bedeutet die Abkürzung EQU?"
                </li>
                <li onClick={() => setQuestion('Welche Einheit wird für Kraft empfohlen?')}>
                  "Welche Einheit wird für Kraft empfohlen?"
                </li>
                <li onClick={() => setQuestion('List all Teilsicherheitsbeiwert symbols')}>
                  "List all Teilsicherheitsbeiwert symbols"
                </li>
              </ul>
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className="message-body">
                <div className="message-content">{msg.content}</div>

                {msg.role === 'assistant' &&
                  msg.tools_used &&
                  msg.tools_used.length > 0 && (
                    <div className="tools-info">
                      <details>
                        <summary>
                          Used {msg.tools_used.length} tool(s)
                        </summary>
                        <ul>
                          {msg.tools_used.map((tool, tidx) => (
                            <li key={tidx}>
                              <strong>{tool.tool}</strong>
                              {tool.arguments &&
                                Object.keys(tool.arguments).length > 0 &&
                                `(${JSON.stringify(tool.arguments)})`}
                            </li>
                          ))}
                        </ul>
                      </details>
                    </div>
                  )}
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className="message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-body">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          ref={inputRef}
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Fragen Sie zum Eurocode... / Ask about Eurocode..."
          disabled={loading}
          className="chat-input"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="chat-send-btn"
        >
          {loading ? '...' : '➤'}
        </button>
      </form>
    </div>
  );
}

export default QAChat;
