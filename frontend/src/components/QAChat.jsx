import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';
import './QAChat.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function stripTagFromInlineMath(content) {
  // Match inline math: single $ ... $ (not preceded/followed by another $)
  return content.replace(
    /(?<!\$)\$(?!\$)([\s\S]*?)(?<!\$)\$(?!\$)/g,
    (match, inner) => '$' + inner.replace(/\\tag\s*\*?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}/g, '') + '$'
  );
}

/**
 * Decode HTML entities that OCR/PDF extraction embeds inside LaTeX formulas.
 * Without this, KaTeX throws "Expected '}', got '&'" on strings like
 * \sum_{\mathrm{i} &gt; 1}  (should be >).
 */
function decodeLatexEntities(content) {
  return stripTagFromInlineMath(content)
    .replace(/\\tag\s*\*?\s*\{[^}]*\}/g, '')
    .replace(/&gt;/g, '>')
    .replace(/&lt;/g, '<')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&le;/g, '\\leq')
    .replace(/&ge;/g, '\\geq')
    .replace(/&ne;/g, '\\neq');
}

/**
 * Custom image renderer — resolves relative /api/images/... paths to the
 * backend base URL so images load correctly regardless of where the frontend
 * is hosted.
 */
function CustomImage({ src, alt, ...props }) {
  // LLM sometimes prepends a hallucinated domain (e.g. https://example.com)
  // to relative /api/images/... paths.  Extract the /api/ path and resolve
  // it against the real backend URL.
  let resolvedSrc = src;
  if (src) {
    const apiMatch = src.match(/\/api\/images\/[^\s)]+/);
    if (apiMatch) {
      resolvedSrc = `${API_BASE_URL}${apiMatch[0]}`;
    } else if (src.startsWith('/api/')) {
      resolvedSrc = `${API_BASE_URL}${src}`;
    }
  }
  return (
    <img
      src={resolvedSrc}
      alt={alt || ''}
      style={{ maxWidth: '100%', height: 'auto', margin: '8px 0', borderRadius: '4px' }}
      {...props}
    />
  );
}

const MARKDOWN_PROPS = {
  remarkPlugins: [remarkGfm, remarkMath],
  rehypePlugins: [[rehypeKatex, { throwOnError: false, strict: false }], rehypeRaw],
  components: { img: CustomImage },
};

/**
 * Strip the MD5 hash prefix and .pdf extension from a stored filename.
 * "47d7bca9_normen handbuch eurocode 8.pdf"  →  "normen handbuch eurocode 8"
 */
function cleanDocName(filename) {
  return filename
    .replace(/^[0-9a-f]{32}_/i, '')
    .replace(/\.pdf$/i, '');
}

function SourceItem({ src, index }) {
  const [open, setOpen] = useState(false);
  const displayName = cleanDocName(src.document);
  const href = `${API_BASE_URL}/api/documents/view/${encodeURIComponent(src.document)}#page=${src.page_number}`;

  return (
    <li>
      <div className="source-header">
        <a href={href} target="_blank" rel="noreferrer">
          <span className="source-name">{displayName}</span>
          {' '}— Seite {src.page_number}
        </a>
        {src.preview && (
          <button
            className="source-toggle"
            onClick={() => setOpen((v) => !v)}
            aria-label={open ? 'Vorschau schließen' : 'Vorschau öffnen'}
          >
            {open ? '▼' : '▶'} Vorschau
          </button>
        )}
      </div>
      {src.chapter && <small>{src.chapter}</small>}
      {open && src.preview && (
        <div className="source-preview">
          <ReactMarkdown {...MARKDOWN_PROPS} >
            {decodeLatexEntities(src.preview)}
          </ReactMarkdown>
        </div>
      )}
    </li>
  );
}

function SourcesPanel({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources">
      <h4>Quellen / Sources</h4>
      <ul>
        {sources.map((src, i) => (
          <SourceItem key={i} src={src} index={i} />
        ))}
      </ul>
    </div>
  );
}

function AssistantMessage({ msg, streamStatus }) {
  const isStreaming = msg.streaming;
  const hasContent = Boolean(msg.content);

  return (
    <div className={`message assistant${isStreaming ? ' streaming' : ''}`}>
      <div className="message-content">
        {/* Show status pill while we have no tokens yet */}
        {isStreaming && !hasContent && (
          <div className="stream-status">
            <span className="stream-status-dot" />
            {streamStatus || 'Verbinde…'}
          </div>
        )}

        {/* Render accumulated / final content */}
        {hasContent && (
          <ReactMarkdown {...MARKDOWN_PROPS}>
            {decodeLatexEntities(msg.content)}
          </ReactMarkdown>
        )}

        {/* Blinking cursor while tokens are still arriving */}
        {isStreaming && hasContent && <span className="typing-cursor" />}
      </div>

      {/* Sources panel — only after the response is complete */}
      {!isStreaming && <SourcesPanel sources={msg.sources} />}

      {/* Tool metadata — only after the response is complete */}
      {!isStreaming && msg.tools_used && msg.tools_used.length > 0 && (
        <div className="tools-info">
          <details>
            <summary>Agent used {msg.tools_used.length} tool(s)</summary>
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
  );
}

function QAChat() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamStatus, setStreamStatus] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamStatus]);

  /** Handle a single parsed SSE event from the stream. */
  const handleStreamEvent = useCallback((event) => {
    if (event.type === 'status') {
      setStreamStatus(event.message);
    } else if (event.type === 'token') {
      // Append the token to the last (streaming) assistant message
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.streaming) {
          return [
            ...prev.slice(0, -1),
            { ...last, content: last.content + event.content },
          ];
        }
        return prev;
      });
    } else if (event.type === 'error') {
      setMessages((prev) => {
        const msgs = [...prev];
        const last = msgs[msgs.length - 1];
        const errMsg = { role: 'assistant', content: `Fehler: ${event.message}`, streaming: false, tools_used: [] };
        if (last?.streaming) return [...msgs.slice(0, -1), errMsg];
        return [...msgs, errMsg];
      });
      setStreamStatus('');
    } else if (event.type === 'done') {
      // Replace the streaming placeholder with the finalised, post-processed answer
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.streaming) {
          return [
            ...prev.slice(0, -1),
            {
              role: 'assistant',
              content: event.answer,
              tools_used: event.tools_used || [],
              sources: event.sources || [],
              streaming: false,
            },
          ];
        }
        return prev;
      });
      setStreamStatus('');
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    const currentQuestion = question;

    // Immediately show user message + empty streaming placeholder
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: currentQuestion },
      { role: 'assistant', content: '', streaming: true, tools_used: [] },
    ]);
    setQuestion('');
    setLoading(true);
    setStreamStatus('Verbinde…');

    try {
      const response = await fetch(`${API_BASE_URL}/api/qa/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: currentQuestion }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by \n\n; keep any incomplete trailing chunk
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() ?? '';

        for (const block of blocks) {
          for (const line of block.split('\n')) {
            if (line.startsWith('data: ')) {
              try {
                handleStreamEvent(JSON.parse(line.slice(6)));
              } catch {
                // malformed JSON — skip
              }
            }
          }
        }
      }
    } catch (error) {
      // Replace streaming placeholder with error message
      setMessages((prev) => {
        const msgs = [...prev];
        const last = msgs[msgs.length - 1];
        const errMsg = { role: 'assistant', content: `Fehler: ${error.message}`, streaming: false, tools_used: [] };
        if (last?.streaming) {
          return [...msgs.slice(0, -1), errMsg];
        }
        return [...msgs, errMsg];
      });
    } finally {
      setLoading(false);
      setStreamStatus('');
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
          messages.map((msg, idx) =>
            msg.role === 'assistant' ? (
              <AssistantMessage key={idx} msg={msg} streamStatus={streamStatus} />
            ) : (
              <div key={idx} className="message user">
                <div className="message-content">{msg.content}</div>
              </div>
            )
          )
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
          {loading ? '…' : '➤'}
        </button>
      </form>
    </div>
  );
}

export default QAChat;
