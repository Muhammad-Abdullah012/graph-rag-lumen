import React, { useState, useRef, useEffect } from 'react';
import './QAChat.css';

function QAChat() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    // Add user message
    const userMessage = { role: 'user', content: question };
    setMessages([...messages, userMessage]);
    setQuestion('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/qa/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          top_k: 5,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get answer');
      }

      const data = await response.json();

      // Add assistant message
      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        confidence: data.confidence,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.message}`,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="qa-chat">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-chat">
            <h2>Ask a Question</h2>
            <p>Ask anything about your uploaded documents</p>
            <div className="example-questions">
              <h3>Example questions:</h3>
              <ul>
                <li>"What is the main topic?"</li>
                <li>"Summarize the document"</li>
                <li>"What are the key points?"</li>
              </ul>
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.content}
              </div>
              
              {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  <h4>Sources:</h4>
                  <ul>
                    {msg.sources.map((source, sidx) => (
                      <li key={sidx}>
                        <a href={source.document_url} target="_blank" rel="noopener noreferrer">
                          {source.document_name}
                        </a>
                        <small>
                          {source.page_number > 0 && `Page ${source.page_number + 1} | `}
                          Relevance: {(source.relevance_score * 100).toFixed(1)}%
                        </small>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {msg.role === 'assistant' && msg.confidence !== undefined && (
                <div className="confidence">
                  Confidence: {(msg.confidence * 100).toFixed(1)}%
                </div>
              )}
            </div>
          ))
        )}
        {loading && (
          <div className="message assistant">
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your documents..."
          disabled={loading}
          className="chat-input"
        />
        <button 
          type="submit" 
          disabled={loading || !question.trim()}
          className="chat-send-btn"
        >
          {loading ? 'ᯓ➤' : '➤'}
        </button>
      </form>
    </div>
  );
}

export default QAChat;
