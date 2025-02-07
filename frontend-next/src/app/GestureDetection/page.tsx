"use client";
import React, { useEffect, useRef, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

interface Gesture {
  handedness: string; 
  gesture: string;
}

const GestureDetection = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [gestures, setGestures] = useState<Gesture[]>([]);
  const [annotatedFrame, setAnnotatedFrame] = useState<string>('');

  useEffect(() => {
    const setupWebSocket = () => {
      const ws = new WebSocket('ws://localhost:8000/ws/gesture');
      
      ws.onopen = () => {
        setIsConnected(true);
        console.log('Connected to WebSocket');
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        console.log('Disconnected from WebSocket');
        setTimeout(setupWebSocket, 2000);
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setGestures(data.gestures || []);
        if (data.processed_frame) {
          setAnnotatedFrame(data.processed_frame);
        }
      };
      
      wsRef.current = ws;
    };

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Error accessing camera:', err);
      }
    };

    setupCamera();
    setupWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    const sendFrame = () => {
      if (!isConnected || !videoRef.current || !canvasRef.current) return;

      const canvas = canvasRef.current;
      const video = videoRef.current;
      const context = canvas.getContext('2d');
      if (!context) return;

      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const base64Frame = canvas.toDataURL('image/jpeg', 0.8);
      if (wsRef.current) {
        wsRef.current.send(base64Frame);
      }
    };

    const intervalId = setInterval(sendFrame, 100); // 10 FPS

    return () => clearInterval(intervalId);
  }, [isConnected]);

  return (
    <div className="p-4">
      <Card className="w-full max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>Gesture Detection</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative">
            {/* Option 1: Show the local video feed */}
            <video
              ref={videoRef}
              className="w-full rounded-lg"
              autoPlay
              playsInline
              style={{ display: annotatedFrame ? 'none' : 'block' }}
            />
            <canvas ref={canvasRef} width={640} height={480} className="hidden" />

            {/* Option 2: Show the annotated frame from backend */}
            {annotatedFrame && (
              <img
                src={annotatedFrame}
                alt="Annotated Gesture Frame"
                className="w-full rounded-lg"
              />
            )}
            
            {/* Overlay for gesture information */}
            <div className="absolute top-4 left-4 bg-black/50 text-white p-2 rounded">
              {gestures.map((gesture, idx) => (
                <div key={idx} className="mb-2">
                  <p className="font-bold">
                    {gesture.handedness} Hand: {gesture.gesture}
                  </p>
                </div>
              ))}
            </div>

            {/* Connection status */}
            <div className={`absolute top-4 right-4 p-2 rounded ${
              isConnected ? 'bg-green-500' : 'bg-red-500'
            } text-white`}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default GestureDetection;
