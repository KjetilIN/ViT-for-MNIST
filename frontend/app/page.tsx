"use client"

import DigitCanvas from "@/components/ui/digitCanvas";
import {useEffect, useState } from "react";

export default function Home() {
  // State of the canvas 
  const [matrix, setMatrix] = useState<number[]>([]);

  // Digit that has been predicted 
  // const [digit, setDigit] = useState<number | null>(null);

  // Backend server IP
  //const backendIP = "http://localhost:3000";
  //const predictApiPath = "/predict";
  //const predictApi = backendIP + predictApiPath; 


  // Use effect to log matrix state when it changes 
  useEffect(() =>{
    console.log(matrix)
    // Do a post request with the matrix list as body to the given api
    

  }, [matrix])

  return (
    <div className="bg-blue-950 h-screen flex justify-center items-center">
      
      <h1 className="text-white font-extrabold text-4xl">Draw a digit</h1>


      <div className="border-lg m-3 border-black h-fit w-fit">
      
        <DigitCanvas setMatrix={setMatrix} />

      </div>
      

    </div>
  );
}
