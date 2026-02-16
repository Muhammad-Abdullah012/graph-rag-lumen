import React, { useState, useRef, useEffect, useCallback } from 'react';
import './QAChat.css';

function QAChat({ conversationId, onFirstMessage }) {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
    if (!question.trim() || !conversationId) return;

    const userMessage = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMessage]);
    const currentQuestion = question;
    setQuestion('');
    setLoading(true);

    // Notify parent about first message (for auto-title)
    if (messages.length === 0 && onFirstMessage) {
      onFirstMessage(currentQuestion);
    }

    try {
      // Use streaming endpoint
      const response = await fetch(
        `${API_BASE_URL}/api/conversations/${conversationId}/chat/stream`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: currentQuestion }),
        }
      );

      if (!response.ok) throw new Error('Failed to get answer');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let assistantContent = '';
      let toolsUsed = [];

      // Add placeholder assistant message
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: '', tools_used: [] },
      ]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const event = JSON.parse(jsonStr);
            if (event.type === 'token') {
              assistantContent += event.content;
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  content: assistantContent,
                };
                return updated;
              });
            } else if (event.type === 'tool') {
              toolsUsed.push({ tool: event.name, arguments: {} });
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  tools_used: [...toolsUsed],
                };
                return updated;
              });
            }
            // 'done' event — streaming complete
          } catch {
            // skip malformed JSON
          }
        }
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev.filter((m) => m.content !== ''),
        { role: 'assistant', content: `Error: ${error.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  if (!conversationId) {
    return (
      <div className="qa-chat">
        <div className="chat-messages">
          <div className="empty-chat">
            <h2>Eurocode Knowledge Q&A</h2>
            <p>Create or select a conversation to start chatting.</p>
          </div>
        </div>
      </div>
    );
  }

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
