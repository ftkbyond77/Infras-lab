"use client";
import { useState } from "react";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // 1. Handle File Selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedImage(e.target.files[0]);
      setResultImage(null); // Clear previous result
    }
  };

  // 2. Handle Upload & Inference
  const handleUpload = async () => {
    if (!selectedImage) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      // Connect to your Docker Backend (Port 8080)
      const response = await fetch("http://localhost:8081/infer", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Inference failed");

      // 3. Convert Response to Image URL
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setResultImage(imageUrl);
    } catch (error) {
      console.error(error);
      alert("Error processing image. Check the backend logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-10 bg-black">
      {/* Vintage Grid Background Pattern */}
      <div className="fixed inset-0 opacity-5 pointer-events-none"
        style={{
          backgroundImage: `linear-gradient(white 1px, transparent 1px), linear-gradient(90deg, white 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        }}
      />

      {/* Header */}
      <div className="relative z-10 text-center mb-12">
        <div className="inline-block border-4 border-white px-8 py-4 bg-black">
          <h1 className="text-5xl font-black tracking-wider text-white" style={{ fontFamily: 'Courier New, monospace' }}>
            HORSE → ZEBRA
          </h1>
          <div className="mt-2 text-sm tracking-widest text-white opacity-70">
            ━━━ AI TRANSFORMATION ENGINE ━━━
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 flex gap-8 items-center">
        {/* Input Section */}
        <div className="flex flex-col items-center">
          <div className="mb-4 border-2 border-white px-4 py-1 bg-black">
            <h2 className="text-sm font-bold tracking-widest text-white" style={{ fontFamily: 'Courier New, monospace' }}>
              [ INPUT ]
            </h2>
          </div>
          
          <div className="relative group">
            <div className="absolute -inset-1 bg-white opacity-0 group-hover:opacity-20 transition-opacity duration-300" />
            <div className="relative w-80 h-80 border-4 border-white bg-black flex items-center justify-center overflow-hidden">
              {selectedImage ? (
                <img 
                  src={URL.createObjectURL(selectedImage)} 
                  className="w-full h-full object-cover"
                  alt="Input horse"
                />
              ) : (
                <div className="text-center p-6">
                  <div className="text-6xl mb-4">🐴</div>
                  <div className="text-white text-sm tracking-wider opacity-50">
                    NO IMAGE LOADED
                  </div>
                </div>
              )}
            </div>
          </div>

          <label className="mt-6 cursor-pointer group">
            <div className="border-2 border-white px-6 py-2 bg-black hover:bg-white hover:text-black transition-all duration-200">
              <span className="text-sm font-bold tracking-widest text-white group-hover:text-black" style={{ fontFamily: 'Courier New, monospace' }}>
                CHOOSE FILE
              </span>
            </div>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleFileChange} 
              className="hidden"
            />
          </label>
        </div>

        {/* Transform Button */}
        <div className="flex flex-col items-center gap-3">
          <button
            onClick={handleUpload}
            disabled={loading || !selectedImage}
            className="border-4 border-white px-8 py-8 bg-black hover:bg-white disabled:opacity-30 disabled:hover:bg-black transition-all duration-200 group"
          >
            {loading ? (
              <div className="flex flex-col items-center gap-2">
                <div className="w-8 h-8 border-4 border-white border-t-transparent animate-spin" />
                <span className="text-xs font-bold tracking-widest text-white" style={{ fontFamily: 'Courier New, monospace' }}>
                  PROCESSING
                </span>
              </div>
            ) : (
              <div className="text-4xl font-black text-white group-hover:text-black transition-colors">
                →
              </div>
            )}
          </button>
          <div className="text-xs tracking-widest text-white opacity-50" style={{ fontFamily: 'Courier New, monospace' }}>
            TRANSFORM
          </div>
        </div>

        {/* Output Section */}
        <div className="flex flex-col items-center">
          <div className="mb-4 border-2 border-white px-4 py-1 bg-black">
            <h2 className="text-sm font-bold tracking-widest text-white" style={{ fontFamily: 'Courier New, monospace' }}>
              [ OUTPUT ]
            </h2>
          </div>
          
          <div className="relative group">
            <div className="absolute -inset-1 bg-white opacity-0 group-hover:opacity-20 transition-opacity duration-300" />
            <div className="relative w-80 h-80 border-4 border-white bg-black flex items-center justify-center overflow-hidden">
              {resultImage ? (
                <img 
                  src={resultImage} 
                  className="w-full h-full object-cover"
                  alt="Output zebra"
                />
              ) : (
                <div className="text-center p-6">
                  <div className="text-6xl mb-4">🦓</div>
                  <div className="text-white text-sm tracking-wider opacity-50">
                    AWAITING TRANSFORMATION
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 border-2 border-white px-6 py-2 bg-black opacity-50">
            <span className="text-sm font-bold tracking-widest text-white" style={{ fontFamily: 'Courier New, monospace' }}>
              RESULT
            </span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="relative z-10 mt-16 text-center">
        <div className="text-xs tracking-widest text-white opacity-30" style={{ fontFamily: 'Courier New, monospace' }}>
          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        </div>
        <div className="mt-2 text-xs tracking-widest text-white opacity-50" style={{ fontFamily: 'Courier New, monospace' }}>
          POWERED BY NEURAL NETWORKS · EST. 2025
        </div>
      </div>
    </main>
  );
}