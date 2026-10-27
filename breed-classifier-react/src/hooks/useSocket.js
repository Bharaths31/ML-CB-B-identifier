import { useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';

export function useSocket() {
  const socketRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [serverStatus, setServerStatus] = useState(null);
  const [predictionEvent, setPredictionEvent] = useState(null);
  const [oodStatus, setOodStatus] = useState(null);
  const [modelSwitched, setModelSwitched] = useState(null);
  const [escalationEvent, setEscalationEvent] = useState(null);

  useEffect(() => {
    // Flask-SocketIO emits on the "/ws" namespace (server/websocket/events.py).
    // Connecting to "/" here silently missed every server event.
    const socket = io('/ws', { path: '/socket.io', transports: ['websocket', 'polling'] });
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
    socket.on('model_switched', (data) => {
        setModelSwitched(data);
        socket.emit('request_status', {});
    });
    socket.on('ood_status_changed', (data) => setOodStatus(data));
    socket.on('escalation_created', (data) => setEscalationEvent({ type: 'created', ...data }));
    socket.on('escalation_resolved', (data) => setEscalationEvent({ type: 'resolved', ...data }));
    socket.on('model_loaded', (data) => {
        // optionally could trigger a status refresh
        socket.emit('request_status', {});
    });

    return () => socket.disconnect();
  }, []);

  return {
    connected,
    serverStatus,
    predictionEvent,
    oodStatus,
    modelSwitched,
    escalationEvent,
    socket: socketRef,
  };
}
