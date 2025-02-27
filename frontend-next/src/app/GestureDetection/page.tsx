"use client";
import React, { useEffect, useRef, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import Image from "next/image";

interface Gesture {
  handedness: string;
  gesture: string;
}

const GestureDetection = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Detection
  const [isConnected, setIsConnected] = useState(false);
  const [gestures, setGestures] = useState<Gesture[]>([]);
  const [annotatedFrame, setAnnotatedFrame] = useState<string>("");

  // Training
  const [isTrainingMode, setIsTrainingMode] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState("");
  const [gestureName, setGestureName] = useState("");
  const trainingFramesRef = useRef<string[]>([]);

  // Camera
  useEffect(() => {
    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };
    setupCamera();
  }, []);

  // WebSocket
  useEffect(() => {
    if (isTrainingMode) return;

    const setupWebSocket = () => {
      const ws = new WebSocket("ws://localhost:8000/ws/gesture");

      ws.onopen = () => {
        setIsConnected(true);
        console.log("Connected to WebSocket");
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log("Disconnected from WebSocket");
        setTimeout(setupWebSocket, 2000);
      };

      ws.onmessage = (event) => {
        const { gestures, processed_frame } = JSON.parse(event.data);
        setGestures(gestures || []);
        if (processed_frame) {
          setAnnotatedFrame(processed_frame);
        }
      };

      wsRef.current = ws;
    };

    setupWebSocket();

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [isTrainingMode]);

  // Capture gest
  useEffect(() => {
    const captureFrame = () => {
      if (!videoRef.current || !canvasRef.current) return;
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const context = canvas.getContext("2d");
      if (!context) return;

      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const base64Frame = canvas.toDataURL("image/jpeg", 0.8);

      if (isTrainingMode) {
        trainingFramesRef.current.push(base64Frame);
      } else if (wsRef.current && isConnected) {
        wsRef.current.send(base64Frame);
      }
    };

    const intervalId = setInterval(captureFrame, 100); // 10 FPS
    return () => clearInterval(intervalId);
  }, [isConnected, isTrainingMode]);

  // Training
  const handleStartTraining = async (mode: "sync" | "async" = "async") => {
    if (!gestureName.trim()) {
      setTrainingStatus("Please enter a gesture name.");
      return;
    }

    trainingFramesRef.current = [];
    setTrainingStatus("Capturing training data...");
    setIsTrainingMode(true);

    setTimeout(async () => {
      setIsTrainingMode(false);
      setTrainingStatus(`Sending ${trainingFramesRef.current.length} frames to backend...`);

      try {
        const endpoint =
          mode === "sync"
            ? "http://localhost:8000/gesture/train/sync"
            : "http://localhost:8000/gesture/train/async";
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            training_images: trainingFramesRef.current,
            gesture_name: gestureName,
            //TODO: Frontend or backend for these below
            learning_rate: 0.001,
            epochs: 5,
            batch_size: 1,
          }),
        });

        const result = await response.json();
        if (response.ok) {
          setTrainingStatus(
            mode === "sync"
              ? `Training finished: ${result.message}`
              : "Training queued successfully."
          );
        } else {
          setTrainingStatus(`Error: ${result.detail || "Training failed"}`);
        }
      } catch (error: unknown) {
        const errMsg = error instanceof Error ? error.message : "Unknown error";
        setTrainingStatus(`Error: ${errMsg}`);
      }
    }, 10000); 
  };

  return (
    <div className="p-4">
      <Card className="w-full max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>Gesture Detection</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <video
              ref={videoRef}
              className="w-full rounded-lg"
              autoPlay
              playsInline
            />
            <canvas ref={canvasRef} width={640} height={480} className="hidden" />
            {annotatedFrame && (
              <Image
                src={annotatedFrame}
                alt="Annotated Gesture Frame"
                className="w-full rounded-lg"
                width={640}
                height={480}
              />
            )}
            <div className="absolute top-4 left-4 bg-black/50 text-white p-2 rounded">
              {gestures.map((gesture, idx) => (
                <div key={idx} className="mb-2">
                  <p className="font-bold">
                    {gesture.handedness} Hand: {gesture.gesture}
                  </p>
                </div>
              ))}
            </div>
            <div
              className={`absolute top-4 right-4 p-2 rounded ${
                isConnected ? "bg-green-500" : "bg-red-500"
              } text-white`}
            >
              {isConnected ? "Connected" : "Disconnected"}
            </div>
            {isTrainingMode && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="bg-black/50 text-white p-4 rounded">
                  <p className="text-xl font-bold">Capturing Training Data...</p>
                </div>
              </div>
            )}
          </div>
          <div className="flex flex-col items-center mt-4">
            <input
              type="text"
              placeholder="Enter gesture name"
              value={gestureName}
              onChange={(e) => setGestureName(e.target.value)}
              className="mb-4 border border-gray-300 p-2 rounded w-full max-w-xs"
            />
            <div className="flex gap-4">
              <button
                onClick={() => handleStartTraining("sync")}
                disabled={isTrainingMode}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
              >
                Train (Sync)
              </button>
              <button
                onClick={() => handleStartTraining("async")}
                disabled={isTrainingMode}
                className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
              >
                Train (Async)
              </button>
            </div>
          </div>
          {trainingStatus && (
            <div className="mt-2 text-center">
              <p>{trainingStatus}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default GestureDetection;