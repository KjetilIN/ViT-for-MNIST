"use client"
import DigitCanvas from "@/components/ui/digitCanvas";
import PredictionPanel from "@/components/ui/predictionPanel";
import { useEffect, useState } from "react";

export default function Home() {
  // State of the canvas
  const [matrix, setMatrix] = useState<number[]>([]);
  // Digit that has been predicted
  const [digit, setDigit] = useState<number | null>(null);
  const [probabilities, setProbabilities] = useState<{[key: number]: number} | null>();
  // Backend server IP
  const backendIP = "http://localhost:5000";
  const predictApiPath = "/predict";
  const predictApi = backendIP + predictApiPath;

  // Use effect to log matrix state when it changes
  useEffect(() => {
    const predictDigit = async () => {
      // Only make the request if we have data in the matrix
      if (matrix.length > 0) {
        try {
          const response = await fetch(predictApi, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ matrix }),
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          console.log("Prediction: ", data.prediction)
          setDigit(data.prediction);
          setProbabilities(data.probabilities);
        } catch (error) {
          console.error('Error predicting digit:', error);
          // Optionally reset the states on error
          setDigit(null);
          setProbabilities(null);
        }
      }
    };

    predictDigit();
  }, [matrix, predictApi]);

  return (
    <div className="bg-blue-950 h-screen flex flex-col gap-10 justify-center items-center">
      <h1 className="text-white font-extrabold text-5xl">Draw a digit</h1>
      <div className="flex gap-3 border-lg m-3 border-black h-fit w-fit">
        <DigitCanvas setMatrix={setMatrix} />
        {probabilities && digit !== null && (
          <PredictionPanel digit={digit} probabilities={probabilities} />
        )}
      </div>
    </div>
  );
}