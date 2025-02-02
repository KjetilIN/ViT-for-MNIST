"use client"
import { Dispatch, SetStateAction, MouseEvent, useEffect, useRef, useState } from "react";

interface DigitCanvasProps {
  setMatrix: Dispatch<SetStateAction<number[]>>
}

export default function DigitCanvas({setMatrix}: DigitCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const scaleFactor = 20;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Start with white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw in black
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 20;  // Increased line width for better digit recognition
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  const startDrawing = (e: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    getMatrix();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setMatrix([]);
  };

  const getMatrix = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Create a temporary canvas for downscaling
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    // Use better image smoothing
    tempCtx.imageSmoothingEnabled = true;
    tempCtx.imageSmoothingQuality = 'high';
    
    // Draw the main canvas onto the smaller one
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
    // Get the pixel data
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    const matrix: number[] = [];

    // Process each pixel
    for (let i = 0; i < 28; i++) {
      for (let j = 0; j < 28; j++) {
        const idx = (i * 28 + j) * 4;
        // Since we're drawing in black on white, we need to invert the values
        // and normalize to 0-1 range
        const value = 1 - (pixels[idx] / 255);  // Using just red channel since it's grayscale
        matrix.push(parseFloat(value.toFixed(4)));
      }
    }

    // Debug output
    console.log("Matrix stats:", {
      length: matrix.length,
      min: Math.min(...matrix),
      max: Math.max(...matrix),
      sample: matrix.slice(0, 10)
    });

    setMatrix(matrix);
  };

  return (
    <div className="">
      <canvas
        ref={canvasRef}
        width={28 * scaleFactor}
        height={28 * scaleFactor}
        className="touch-none"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
      />
      <div className="flex space-x-4 mt-4 justify-center">
        <button
          onClick={clearCanvas}
          className="bg-gray-700 p-2 rounded-lg text-white hover:bg-gray-800"
        >
          Clear
        </button>
      </div>
    </div>
  );
}