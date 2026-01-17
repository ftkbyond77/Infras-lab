package main

import (
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"runtime"

	"github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

const (
	modelPath   = "../model-training/cyclegan_horse2zebra.onnx"
	inputImage  = "../model-training/data/testA/n02381460_490.jpg"
	outputImage = "zebra_output.jpg"
	imgSize     = 256
)

func main() {
	// 1. Initialize ONNX Runtime
	var libName string
	if runtime.GOOS == "windows" {
		libName = "onnxruntime.dll" // For Windows
	} else {
		libName = "libonnxruntime.so" // For Linux
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
		[]string{"input"},  // Matches Python input name
		[]string{"output"}, // Matches Python output name
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	defer session.Destroy()

	// 3. Prepare Input Data
	log.Printf("Processing image: %s", inputImage)
	inputTensorData, err := preprocessImage(inputImage, imgSize, imgSize)
	if err != nil {
		log.Fatalf("Preprocessing failed: %v", err)
	}

	// [FIX 1] Use Generic Instantiation [float32]
	shape := onnxruntime_go.NewShape(1, 3, int64(imgSize), int64(imgSize))
	inputTensor, err := onnxruntime_go.NewTensor[float32](shape, inputTensorData)
	if err != nil {
		log.Fatal(err)
	}
	defer inputTensor.Destroy()

	// 4. Run Inference
	log.Println("Running inference...")

	// [FIX 2] Create inputs and outputs slices
	// We pass a slice of inputs and an empty slice for outputs (library will fill it)
	inputs := []onnxruntime_go.Value{inputTensor}
	outputs := make([]onnxruntime_go.Value, 1) // Reserve slot for 1 output

	// [FIX 3] Run accepts (inputs, outputs) and returns only error
	err = session.Run(inputs, outputs)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	// 5. Get Output
	// The output is now in outputs[0]. We must type assert it to the generic Tensor type.
	outputTensor := outputs[0].(*onnxruntime_go.Tensor[float32])
	defer outputTensor.Destroy() // Don't forget to cleanup the library-allocated tensor

	outputData := outputTensor.GetData()

	// 6. Post-process and Save
	err = postprocessAndSave(outputData, imgSize, imgSize, outputImage)
	if err != nil {
		log.Fatalf("Saving failed: %v", err)
	}

	log.Printf("Success! Output saved to %s", outputImage)
}

// preprocessImage loads, resizes, and normalizes the image to [-1, 1]
func preprocessImage(path string, width, height int) ([]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.CatmullRom.Scale(dst, dst.Rect, img, img.Bounds(), draw.Over, nil)

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
	return tensor, nil
}

// postprocessAndSave denormalizes from [-1, 1] to [0, 255] and saves image
func postprocessAndSave(data []float32, width, height int, filename string) error {
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

	out, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer out.Close()

	if filepath.Ext(filename) == ".png" {
		return png.Encode(out, img)
	}
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
