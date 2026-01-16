package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/yalue/onnxruntime_go"
)

func main() {
	// Initialize ONNX Runtime
	onnxruntime_go.SetSharedLibraryPath("path/to/onnxruntime.so") // or .dll on Windows
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		log.Fatal(err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// Create Session
	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		"cyclegan_horse2zebra.onnx",
		[]string{"input"},  // Matches input_names in Python export
		[]string{"output"}, // Matches output_names in Python export
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Destroy()

	// Create dummy input data (1, 3, 256, 256)
	// In a real app, you would load an image and normalize it here
	inputData := make([]float32, 1*3*256*256)
	for i := range inputData {
		inputData[i] = rand.Float32() // Replace with actual normalized pixel data
	}

	// Create Tensor
	inputTensor, err := onnxruntime_go.NewTensor(
		onnxruntime_go.NewShape(1, 3, 256, 256),
		inputData,
	)
	if err != nil {
		log.Fatal(err)
	}
	defer inputTensor.Destroy()

	// Run Inference
	outputTensors, err := session.Run([]*onnxruntime_go.Tensor{inputTensor})
	if err != nil {
		log.Fatal(err)
	}
	defer outputTensors[0].Destroy()

	// Get Output
	outputData := outputTensors[0].GetData()
	fmt.Printf("Inference successful! Output size: %d\n", len(outputData))
	// Convert outputData back to image ...
}