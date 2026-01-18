package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"runtime"

	_ "golang.org/x/image/webp"

	"github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

const (
	modelPath    = "../model-path/cyclegan_horse2zebra.onnx"
	inputImage   = "The-Horses-Personality.jpg"
	outputImage  = "zebra_output.jpg"
	compareImage = "compare.jpg" 
	imgSize      = 256
)

func main() {
	// 1. Initialize ONNX Runtime
	var libName string
	if runtime.GOOS == "windows" {
		libName = "onnxruntime.dll"
	} else {
		libName = "libonnxruntime.so"
	}

	onnxruntime_go.SetSharedLibraryPath(libName)
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX environment: %v\nMake sure %s is in the current folder.", err, libName)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// 2. Load the Model
	log.Println("Loading model...")
	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"output"},
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	defer session.Destroy()

	// 3. Prepare Input Data
	log.Printf("Processing image: %s", inputImage)
	// Modified to return the resized image object as well
	inputTensorData, resizedInputImg, err := preprocessImage(inputImage, imgSize, imgSize)
	if err != nil {
		log.Fatalf("Preprocessing failed: %v", err)
	}

	shape := onnxruntime_go.NewShape(1, 3, int64(imgSize), int64(imgSize))
	inputTensor, err := onnxruntime_go.NewTensor[float32](shape, inputTensorData)
	if err != nil {
		log.Fatal(err)
	}
	defer inputTensor.Destroy()

	// 4. Run Inference
	log.Println("Running inference...")
	inputs := []onnxruntime_go.Value{inputTensor}
	outputs := make([]onnxruntime_go.Value, 1)

	err = session.Run(inputs, outputs)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	// 5. Get Output Data
	outputTensor := outputs[0].(*onnxruntime_go.Tensor[float32])
	defer outputTensor.Destroy()
	outputData := outputTensor.GetData()

	// 6. Post-process (Convert Tensor -> Image)
	outputImg := tensorToImage(outputData, imgSize, imgSize)

	// Save the single output file
	if err := saveImage(outputImg, outputImage); err != nil {
		log.Fatalf("Saving output failed: %v", err)
	}
	log.Printf("Saved output to %s", outputImage)

	// 7. Create and Save Comparison Image (Side-by-Side)
	log.Println("Creating comparison image...")
	
	// Create a new image twice as wide
	comparisonRect := image.Rect(0, 0, imgSize*2, imgSize)
	comparisonImg := image.NewRGBA(comparisonRect)

	// Draw Original Input on the Left
	draw.Draw(comparisonImg, image.Rect(0, 0, imgSize, imgSize), resizedInputImg, image.Point{}, draw.Src)

	// Draw Generated Output on the Right
	draw.Draw(comparisonImg, image.Rect(imgSize, 0, imgSize*2, imgSize), outputImg, image.Point{}, draw.Src)

	// Save the comparison file
	if err := saveImage(comparisonImg, compareImage); err != nil {
		log.Fatalf("Saving comparison failed: %v", err)
	}

	log.Printf("Saved comparison to %s", compareImage)
}

// preprocessImage returns the tensor data AND the resized image object
func preprocessImage(path string, width, height int) ([]float32, image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, nil, fmt.Errorf("decode error: %v", err)
	}

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

// tensorToImage converts raw output data back to a Go Image
func tensorToImage(data []float32, width, height int) *image.RGBA {
	rect := image.Rect(0, 0, width, height)
	img := image.NewRGBA(rect)
	channelStride := width * height

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := y*width + x

			// Denormalize: (val * 0.5) + 0.5
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

// saveImage is a utility to save an image object to disk
func saveImage(img image.Image, filename string) error {
	out, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer out.Close()

	if filepath.Ext(filename) == ".png" {
		return png.Encode(out, img)
	}
	// Use default JPEG quality
	return jpeg.Encode(out, img, nil)
}

func clamp(val float32) float32 {
	if val < 0 {
		return 0
	}
	if val > 1 {
		return 1
	}
	return val
}