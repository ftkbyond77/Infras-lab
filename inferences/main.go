package main

import (
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"net/http"
	"os"
	"runtime"

	_ "image/png" // Register PNG decoder
	_ "golang.org/x/image/webp"

	"github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

var (
	// We load the model once, globally
	session *onnxruntime_go.DynamicAdvancedSession
	imgSize = 256
)

func main() {
	// 1. Initialize ONNX Runtime
	initONNX()

	// 2. Setup HTTP Server
	http.HandleFunc("/infer", handleInference)
	
	// Health check for Render
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Server listening on port %s...", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func initONNX() {
	var libName string
	if runtime.GOOS == "windows" {
		libName = "onnxruntime.dll"
	} else {
		libName = "libonnxruntime.so"
	}

	// Ensure library is loaded
	onnxruntime_go.SetSharedLibraryPath(libName)
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX environment: %v", err)
	}

	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "../model-path/cyclegan_horse2zebra.onnx"
	}

	log.Println("Loading model into memory...")
	session, err = onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"output"},
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	log.Println("Model loaded successfully.")
}

func handleInference(w http.ResponseWriter, r *http.Request) {
	// CORS Headers (Enable access from Next.js)
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST")
	
	if r.Method == "OPTIONS" {
		return
	}
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 1. Parse Image from Form Data
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "Invalid file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	srcImg, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "Failed to decode image", http.StatusBadRequest)
		return
	}

	// 2. Preprocess
	inputTensorData, _, err := preprocessImage(srcImg, imgSize, imgSize)
	if err != nil {
		http.Error(w, "Preprocessing failed", http.StatusInternalServerError)
		log.Printf("Preprocess error: %v", err)
		return
	}

	// 3. Create Tensor
	shape := onnxruntime_go.NewShape(1, 3, int64(imgSize), int64(imgSize))
	inputTensor, err := onnxruntime_go.NewTensor[float32](shape, inputTensorData)
	if err != nil {
		http.Error(w, "Tensor creation failed", http.StatusInternalServerError)
		return
	}
	defer inputTensor.Destroy()

	// 4. Run Inference
	inputs := []onnxruntime_go.Value{inputTensor}
	outputs := make([]onnxruntime_go.Value, 1)

	err = session.Run(inputs, outputs)
	if err != nil {
		http.Error(w, "Inference failed", http.StatusInternalServerError)
		log.Printf("Inference error: %v", err)
		return
	}

	// 5. Post-process
	outputTensor := outputs[0].(*onnxruntime_go.Tensor[float32])
	defer outputTensor.Destroy()
	outputImg := tensorToImage(outputTensor.GetData(), imgSize, imgSize)

	// 6. Return Image Response directly
	w.Header().Set("Content-Type", "image/jpeg")
	if err := jpeg.Encode(w, outputImg, nil); err != nil {
		log.Printf("Response write error: %v", err)
	}
}

func preprocessImage(img image.Image, width, height int) ([]float32, image.Image, error) {
	// Resize using high-quality resampling
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.CatmullRom.Scale(dst, dst.Rect, img, img.Bounds(), draw.Over, nil)

	// Convert to Tensor (CHW format)
	tensor := make([]float32, 3*width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := dst.At(x, y).RGBA()

			// Normalize to [-1, 1]
			rNorm := (float32(r)/65535.0 - 0.5) / 0.5
			gNorm := (float32(g)/65535.0 - 0.5) / 0.5
			bNorm := (float32(b)/65535.0 - 0.5) / 0.5

			idx := y*width + x
			tensor[idx] = rNorm                  // R
			tensor[idx+(width*height)] = gNorm   // G
			tensor[idx+(2*width*height)] = bNorm // B
		}
	}
	return tensor, dst, nil
}

func tensorToImage(data []float32, width, height int) *image.RGBA {
	rect := image.Rect(0, 0, width, height)
	img := image.NewRGBA(rect)
	channelStride := width * height

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := y*width + x

			r := (data[idx] * 0.5) + 0.5
			g := (data[idx+channelStride] * 0.5) + 0.5
			b := (data[idx+(2*channelStride)] * 0.5) + 0.5

			img.Set(x, y, color.RGBA{
				R: uint8(clamp(r) * 255),
				G: uint8(clamp(g) * 255),
				B: uint8(clamp(b) * 255),
				A: 255,
			})
		}
	}
	return img
}

func clamp(val float32) float32 {
	if val < 0 { return 0 }
	if val > 1 { return 1 }
	return val
}