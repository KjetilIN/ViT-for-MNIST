"use client"
import { Dispatch, SetStateAction, MouseEvent, useEffect, useRef, useState } from "react";

interface DigitCanvasProps {
  setMatrix: Dispatch<SetStateAction<number[]>>
}

export default function DigitCanvas({setMatrix}: DigitCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const CANVAS_SIZE = 28;
  const DISPLAY_SCALE = 20;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set actual canvas size to 28x28
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    
    // Black background
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    
    // White drawing color
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  const getCanvasMousePosition = (e: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) * (CANVAS_SIZE / rect.width));
    const y = Math.floor((e.clientY - rect.top) * (CANVAS_SIZE / rect.height));
    return { x, y };
  };

  const startDrawing = (e: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { x, y } = getCanvasMousePosition(e);
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
    const { x, y } = getCanvasMousePosition(e);
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
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setMatrix([]);
  };

  const getMatrix = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const pixels = imageData.data;
    const matrix: number[] = [];

    // Convert to grayscale values 0-255
    for (let i = 0; i < CANVAS_SIZE; i++) {
      for (let j = 0; j < CANVAS_SIZE; j++) {
        const idx = (i * CANVAS_SIZE + j) * 4;
        // Taking just the red channel since it's grayscale anyway
        const value = pixels[idx];
        matrix.push(value);
      }
    }

    // Debug visualization in console
    console.log("\nDigit Visualization:");
    for (let i = 0; i < CANVAS_SIZE; i++) {
      let row = '';
      for (let j = 0; j < CANVAS_SIZE; j++) {
        row += matrix[i * CANVAS_SIZE + j] > 127 ? '#' : '.';
      }
      console.log(row);
    }

    console.log("Matrix stats:", {
      length: matrix.length,
      min: Math.min(...matrix),
      max: Math.max(...matrix)
    });

    setMatrix(matrix);
  };

  return (
    <div className="">
      <canvas
        ref={canvasRef}
        style={{
          width: `${CANVAS_SIZE * DISPLAY_SCALE}px`,
          height: `${CANVAS_SIZE * DISPLAY_SCALE}px`,
          imageRendering: 'pixelated'
        }}
        className="touch-none border border-gray-300"
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