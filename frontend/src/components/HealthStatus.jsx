import React, { useState, useEffect } from 'react';
import './HealthStatus.css';

function HealthStatus() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/health/`);
        if (response.ok) {
          const data = await response.json();
          setHealth(data);
        }
      } catch (error) {
        console.error('Health check failed:', error);
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
    // const interval = setInterval(checkHealth, 15000);
    // return () => clearInterval(interval);
  }, [API_BASE_URL]);

  if (loading) return <div className="health-status">Checking...</div>;

  const getStatusColor = (status) => (status === 'ok' ? 'green' : 'red');

  return (
    <div className="health-status">
      <div className={`status-badge ${getStatusColor(health?.status)}`}>
        {health?.status === 'ok' ? '✅' : '⚠️'} {health?.status || 'unknown'}
      </div>
      <div className="health-details">
        <span>Neo4j: <strong>{health?.neo4j || '?'}</strong></span>
        <span>Postgres: <strong>{health?.postgres || '?'}</strong></span>
        <span>Ollama: <strong>{health?.ollama || '?'}</strong></span>
        <span>Graph: <strong>{health?.graph_loaded ? 'loaded' : 'empty'}</strong></span>
      </div>
    </div>
  );
}

export default HealthStatus;
