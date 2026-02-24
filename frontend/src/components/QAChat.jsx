import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';
import './QAChat.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Custom image renderer — resolves relative /api/images/... paths to the
 * backend base URL so images load correctly regardless of where the frontend
 * is hosted.
 */
function CustomImage({ src, alt, ...props }) {
  const resolvedSrc =
    src && src.startsWith('/api/')
      ? `${API_BASE_URL}${src}`
      : src;
  return (
    <img
      src={resolvedSrc}
      alt={alt || ''}
      style={{ maxWidth: '100%', height: 'auto', margin: '8px 0', borderRadius: '4px' }}
      {...props}
    />
  );
}

function QAChat() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const userMessage = { role: 'user', content: question };
    setMessages(prev => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/qa/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) throw new Error('Failed to get answer');

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        tools_used: data.tools_used || [],
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Fehler: ${error.message}`,
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="qa-chat">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-chat">
            <h2>Frage stellen / Ask a Question</h2>
            <p>Stellen Sie Fragen zum Eurocode-Wissensgraphen</p>
            <div className="example-questions">
              <h3>Beispielfragen:</h3>
              <ul>
                <li onClick={() => setQuestion("Wie kann der Bemessungswert der Tragfähigkeit ausgedrückt werden?")}>
                  "Wie kann der Bemessungswert der Tragfähigkeit ausgedrückt werden?"
                </li>
                <li onClick={() => setQuestion("Was bedeutet das Symbol γf?")}>
                  "Was bedeutet das Symbol γf?"
                </li>
                <li onClick={() => setQuestion("Wie lautet die Formel für AEd?")}>
                  "Wie lautet die Formel für AEd?"
                </li>
                <li onClick={() => setQuestion("Was bedeutet die Abkürzung EQU?")}>
                  "Was bedeutet die Abkürzung EQU?"
                </li>
                <li onClick={() => setQuestion("Welche Einheit wird für Kraft empfohlen?")}>
                  "Welche Einheit wird für Kraft empfohlen?"
                </li>
                <li onClick={() => setQuestion("List all Teilsicherheitsbeiwert symbols")}>
                  "List all Teilsicherheitsbeiwert symbols"
                </li>
              </ul>
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.role === 'assistant' ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[
                      [rehypeKatex, { throwOnError: false, strict: false }],
                      rehypeRaw,
                    ]}
                    components={{
                      img: CustomImage,
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>

              {msg.role === 'assistant' && msg.tools_used && msg.tools_used.length > 0 && (
                <div className="tools-info">
                  <details>
                    <summary>
                      Agent used {msg.tools_used.length} tool(s)
                    </summary>
                    <ul>
                      {msg.tools_used.map((tool, tidx) => (
                        <li key={tidx}>
                          <strong>{tool.tool}</strong>({JSON.stringify(tool.arguments)})
                        </li>
                      ))}
                    </ul>
                  </details>
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
