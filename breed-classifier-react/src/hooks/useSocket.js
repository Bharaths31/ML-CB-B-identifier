import { useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';

export function useSocket() {
  const socketRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [serverStatus, setServerStatus] = useState(null);
  const [predictionEvent, setPredictionEvent] = useState(null);

  useEffect(() => {
    // Connect to same host/port, the vite proxy handles /socket.io
    const socket = io('/', { path: '/socket.io', transports: ['websocket', 'polling'] });
    socketRef.current = socket;

    socket.on('connect', () => {
        setConnected(true);
        socket.emit('request_status', {});
    });
    
    socket.on('disconnect', () => setConnected(false));
    socket.on('server_status', (data) => setServerStatus(data));
    socket.on('prediction_started', (data) => setPredictionEvent({ type: 'started', ...data }));
    socket.on('prediction_progress', (data) => setPredictionEvent({ type: 'progress', ...data }));
    socket.on('prediction_complete', (data) => setPredictionEvent({ type: 'complete', ...data }));
    socket.on('prediction_error', (data) => setPredictionEvent({ type: 'error', ...data }));
    socket.on('model_loaded', (data) => {
        // optionally could trigger a status refresh
        socket.emit('request_status', {});
    });

    return () => socket.disconnect();
  }, []);

  return { connected, serverStatus, predictionEvent, socket: socketRef };
}
