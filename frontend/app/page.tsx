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
      const response = await fetch("http://localhost:8080/infer", {
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
    <main className="min-h-screen flex flex-col items-center justify-center p-10 bg-gray-50 text-black">
      <h1 className="text-3xl font-bold mb-8 text-black">Horse 2 Zebra Generator</h1>

      <div className="flex gap-10 items-start">
        {/* Input */}
        <div className="flex flex-col items-center">
          <h2 className="mb-2 font-semibold">Input (Horse)</h2>
          <input type="file" accept="image/*" onChange={handleFileChange} className="mb-4 text-black"/>
          {selectedImage && (
            <img src={URL.createObjectURL(selectedImage)} className="w-64 h-64 object-cover rounded shadow-lg"/>
          )}
        </div>

        {/* Button */}
        <div className="self-center">
          <button
            onClick={handleUpload}
            disabled={loading || !selectedImage}
            className="bg-blue-600 text-white px-6 py-3 rounded-full hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? "Processing..." : "Transform ->"}
          </button>
        </div>

        {/* Output */}
        <div className="flex flex-col items-center">
          <h2 className="mb-2 font-semibold">Result (Zebra)</h2>
          <div className="flex items-center justify-center w-64 h-64 border-2 border-dashed border-gray-300 rounded bg-white">
            {resultImage ? (
              <img src={resultImage} className="w-full h-full object-cover rounded shadow-lg"/>
            ) : (
              <span className="text-gray-400">Result here</span>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

