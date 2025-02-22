"use client"
import { useState, useEffect } from "react";

export default function GestureTrainingPage() {
  const [gestureName, setGestureName] = useState("");
  const [instructions, setInstructions] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [trainStatus, setTrainStatus] = useState("");

  const fetchInstructions = async () => {
    try {
      const response = await fetch("http://localhost:8000/instructions");
      const data = await response.json();
      setInstructions(data.instructions);
    } catch (error) {
      console.error("Error fetching instructions:", error);
    }
  };

  const handleImageUpload = async () => {
    if (!gestureName || !selectedFile) {
      alert("Please enter a gesture name and select an image.");
      return;
    }
    const formData = new FormData();
    formData.append("gesture_name", gestureName);
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/capture", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setUploadStatus(data.message);
    } catch (error) {
      console.error("Error uploading image:", error);
    }
  };

  const triggerTraining = async () => {
    const payload = {
      learning_rate: 0.001,
      epochs: 30,
      batch_size: 1,
      validation_batch_size: 1,
      export_dir: "./exported_models",
      export_model_name: "gesture_recognizer",
    };

    try {
      const response = await fetch("http://localhost:8000/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      setTrainStatus(data.message);
    } catch (error) {
      console.error("Error triggering training:", error);
    }
  };

  // Load instructions on page load.
  useEffect(() => {
    fetchInstructions();
  }, []);

  return (
    <div style={{ padding: "2rem", fontFamily: "Arial, sans-serif" }}>
      <h1>Gesture Recognizer Training</h1>

      <div style={{ marginBottom: "1rem" }}>
        <label htmlFor="gestureName" style={{ marginRight: "1rem" }}>
          Gesture Name:
        </label>
        <input
          type="text"
          id="gestureName"
          value={gestureName}
          onChange={(e) => setGestureName(e.target.value)}
        />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <h2>Capture Instructions</h2>
        <ul>
          {instructions.map((instruction, index) => (
            <li key={index}>{instruction}</li>
          ))}
        </ul>
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <h2>Upload Gesture Image</h2>
        <input
          type="file"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              setSelectedFile(e.target.files[0]);
            }
          }}
        />
        <button onClick={handleImageUpload} style={{ marginLeft: "1rem" }}>
          Upload
        </button>
        {uploadStatus && <p>{uploadStatus}</p>}
      </div>

      <div>
        <button onClick={triggerTraining}>Start Training</button>
        {trainStatus && <p>{trainStatus}</p>}
      </div>
    </div>
  );
}
