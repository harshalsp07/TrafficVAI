import { useState, useEffect, useRef } from 'react';

export function useSSE(endpoint) {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const esRef = useRef(null);

  useEffect(() => {
    try {
      const es = new EventSource(`/api${endpoint}`);
      esRef.current = es;

      es.onopen = () => setConnected(true);

      const handleEvent = (e) => {
        try {
          const data = JSON.parse(e.data);
          setEvents(prev => [data, ...prev].slice(0, 100));
        } catch {}
      };

      es.onmessage = handleEvent;
      es.addEventListener('violation', handleEvent);
      es.addEventListener('progress', handleEvent);

      es.onerror = () => {
        setConnected(false);
      };

      return () => {
        es.removeEventListener('violation', handleEvent);
        es.removeEventListener('progress', handleEvent);
        es.close();
        setConnected(false);
      };
    } catch {
      setConnected(false);
    }
  }, [endpoint]);

  return { events, connected };
}
