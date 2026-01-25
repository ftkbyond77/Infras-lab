package main

import (
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"net/http"
	"os"
	"runtime"
	"sync" 

	_ "image/png"
	_ "golang.org/x/image/webp"

	"github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

var (
	// Mutex to prevent inference while reloading model
	modelMu sync.RWMutex
	session *onnxruntime_go.DynamicAdvancedSession
	imgSize = 256
)

func main() {
	// 1. Initialize ONNX Runtime Environment (Do this only once)
	initONNXEnvironment()

	// 2. Load the initial model
	loadModelSession()

	// 3. Setup HTTP Server
	http.HandleFunc("/infer", handleInference)
	
	// [NEW] Endpoint for Airflow to trigger model reload
	http.HandleFunc("/reload", handleReload)

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

func initONNXEnvironment() {
	if onnxruntime_go.IsInitialized() {
		return
	}

	var libName string
	if runtime.GOOS == "windows" {
		libName = "onnxruntime.dll"
	} else {
		libName = "libonnxruntime.so"
	}

	onnxruntime_go.SetSharedLibraryPath(libName)
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX environment: %v", err)
	}
}

func loadModelSession() {
	modelMu.Lock() // [LOCK] Block all readers (inference)
	defer modelMu.Unlock()

	// Clean up old session if it exists (Free memory)
	if session != nil {
		session.Destroy()
		session = nil
	}

	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "../model-path/cyclegan_horse2zebra.onnx"
	}

	log.Printf("Loading model from: %s", modelPath)
	var err error
	session, err = onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"output"},
		nil,
	)
	if err != nil {
		log.Printf("ERROR: Failed to create session: %v", err)
	} else {
		log.Println("SUCCESS: Model loaded.")
	}
}

// Handle Reload Request from Airflow
func handleReload(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	log.Println("Received Reload Signal from Pipeline...")
	loadModelSession()
	w.Write([]byte("Model Reloaded Successfully"))
}

func handleInference(w http.ResponseWriter, r *http.Request) {
	// CORS Headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// [READ LOCK] Allow multiple inferences, but block if reloading
	modelMu.RLock()
	defer modelMu.RUnlock()

	if session == nil {
		http.Error(w, "Model is currently reloading or unavailable", http.StatusServiceUnavailable)
		return
	}

	// 1. Parse Image
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

	// 6. Return Image
	w.Header().Set("Content-Type", "image/jpeg")
	if err := jpeg.Encode(w, outputImg, nil); err != nil {
		log.Printf("Response write error: %v", err)
	}
}

func preprocessImage(img image.Image, width, height int) ([]float32, image.Image, error) {
    dst := image.NewRGBA(image.Rect(0, 0, width, height))
    draw.CatmullRom.Scale(dst, dst.Rect, img, img.Bounds(), draw.Over, nil)
    tensor := make([]float32, 3*width*height)
    for y := 0; y < height; y++ {
        for x := 0; x < width; x++ {
            r, g, b, _ := dst.At(x, y).RGBA()
            rNorm := (float32(r)/65535.0 - 0.5) / 0.5
            gNorm := (float32(g)/65535.0 - 0.5) / 0.5
            bNorm := (float32(b)/65535.0 - 0.5) / 0.5
            idx := y*width + x
            tensor[idx] = rNorm
            tensor[idx+(width*height)] = gNorm
            tensor[idx+(2*width*height)] = bNorm
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