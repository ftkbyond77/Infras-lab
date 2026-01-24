package main

import (
	"image"
	"image/color"
	"os"
	"testing"

	"github.com/yalue/onnxruntime_go"
)

// UNIT TEST 1: Preprocessing Logic
// --------------------------------
// Test if a solid White Image is correctly normalized to 1.0
// Logic: White pixel (255) -> (1.0 - 0.5) / 0.5 = 1.0
func TestPreprocessImage_White(t *testing.T) {
	width, height := 256, 256

	// Create a solid WHITE image
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	white := color.RGBA{255, 255, 255, 255}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, white)
		}
	}

	// Run function
	tensor, _, err := preprocessImage(img, width, height)
	if err != nil {
		t.Fatalf("Preprocess failed: %v", err)
	}

	// Validation
	expectedLen := 3 * width * height
	if len(tensor) != expectedLen {
		t.Errorf("Expected tensor length %d, got %d", expectedLen, len(tensor))
	}

	// Check pixel value (Should be close to 1.0)
	if tensor[0] < 0.99 || tensor[0] > 1.01 {
		t.Errorf("Normalization error. Expected ~1.0 for white pixel, got %f", tensor[0])
	}
}


// UNIT TEST 2: Quality/Post-processing Logic
// ------------------------------------------
// Test if a Tensor value of 1.0 gets converted back to ~255 (White)
func TestTensorToImage(t *testing.T) {
	width, height := 256, 256
	tensorData := make([]float32, 3*width*height)

	// Fill tensor with 1.0
	for i := range tensorData {
		tensorData[i] = 1.0
	}

	outputImg := tensorToImage(tensorData, width, height)

	// Check center pixel
	r, g, b, _ := outputImg.At(128, 128).RGBA()

	// RGBA() return uint32 in range [0-0xffff], divide by 257 to get 0-255
	if (r>>8) < 250 {
		t.Errorf("Post-process error. Expected White pixel (>250), got R=%d", r>>8)
	}
	if (g>>8) < 250 {
		t.Errorf("Post-process error. Expected White pixel (>250), got G=%d", g>>8)
	}
	if (b>>8) < 250 {
		t.Errorf("Post-process error. Expected White pixel (>250), got B=%d", b>>8)
	}
}


// INTEGRATION TEST: Full Inference Flow
// ----------------------------------------------------------------------
// This requires a real or dummy ONNX model to be present.
// In CI, we will generate a 'dummy.onnx' before running this.
func TestInferenceFlow(t *testing.T) {
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		t.Skip("Skipping inference test: MODEL_PATH not set")
	}

	// 1. Initialize ONNX (Must find libonnxruntime.so)
	// Note: We use the SharedLibraryPath from initONNX logic if needed, 
	// or assume the test runner has set LD_LIBRARY_PATH.
	onnxruntime_go.SetSharedLibraryPath("libonnxruntime.so")
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		t.Fatalf("Failed to init ONNX: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// 2. Load Model
	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"output"},
		nil,
	)
	if err != nil {
		t.Fatalf("Failed to load model from %s: %v", modelPath, err)
	}
	defer session.Destroy()

	// 3. Create Dummy Input Tensor
	inputData := make([]float32, 1*3*256*256)
	shape := onnxruntime_go.NewShape(1, 3, 256, 256)
	inputTensor, _ := onnxruntime_go.NewTensor[float32](shape, inputData)
	defer inputTensor.Destroy()

	// 4. Run Inference
	inputs := []onnxruntime_go.Value{inputTensor}
	outputs := make([]onnxruntime_go.Value, 1)

	err = session.Run(inputs, outputs)
	if err != nil {
		t.Fatalf("Inference execution failed: %v", err)
	}

	// 5. Verify Output Shape
	outputTensor := outputs[0].(*onnxruntime_go.Tensor[float32])
	outputShape := outputTensor.GetShape()
	
	if outputShape[2] != 256 || outputShape[3] != 256 {
		t.Errorf("Unexpected output shape: %v", outputShape)
	}
}
