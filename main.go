package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

func main() {
	sizes := []int{1000, 5000, 10000, 20000}

	fmt.Println("Time Complexity Benchmark (Go)")
	fmt.Println("--------------------------------------")

	for _, n := range sizes {
		fmt.Printf("\nInput size: %d\n", n)

		data := generateArray(n)

		benchmark("O(1)", func() {
			o1(data)
		})

		benchmark("O(log n)", func() {
			oLogN(data)
		})

		benchmark("O(n)", func() {
			oN(data)
		})

		benchmark("O(n log n)", func() {
			oNLogN(data)
		})

		benchmark("O(n²)", func() {
			oNSquared(n)
		})
	}
}

func benchmark(name string, fn func()) {
	start := time.Now()
	fn()
	elapsed := time.Since(start)
	fmt.Printf("%-12s : %v\n", name, elapsed)
}

// O(1)
func o1(arr []int) int {
	return arr[0]
}

// O(log n) - Binary Search
func oLogN(arr []int) {
	sort.Ints(arr)
	target := arr[len(arr)/2]

	left, right := 0, len(arr)-1
	for left <= right {
		mid := (left + right) / 2
		if arr[mid] == target {
			return
		} else if arr[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
}


// O(n)
func oN(arr []int) int {
	sum := 0
	for _, v := range arr {
		sum += v
	}
	return sum
}


// O(n log n)
func oNLogN(arr []int) {
	sort.Ints(arr)
}

// O(n²)
func oNSquared(n int) {
	count := 0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			count++
		}
	}
}

func generateArray(n int) []int {
	rand.Seed(time.Now().UnixNano())
	arr := make([]int, n)
	for i := 0; i < n; i++ {
		arr[i] = rand.Intn(math.MaxInt32)
	}
	return arr
}
