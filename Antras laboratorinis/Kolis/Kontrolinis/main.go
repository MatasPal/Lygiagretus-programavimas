package main

import (
	"fmt"
	"sync"
)

func senderProcess(id int, ch chan<- int, wg *sync.WaitGroup) {
	defer wg.Done()

	for i := 0; i < 10; i++ {
		ch <- id*10 + i
	}
}

func receiverProcess(ch <-chan int, evenCh, oddCh chan<- int, done chan<- bool, wg *sync.WaitGroup) {
	defer wg.Done()

	count := 0
	for num := range ch {
		count++
		if num%2 == 0 {
			evenCh <- num
		} else {
			oddCh <- num
		}

		if count == 20 {
			close(evenCh)
			close(oddCh)
			done <- true
			break
		}
	}
}

func printerProcess(id int, ch <-chan int, result []int, wg *sync.WaitGroup) {
	defer wg.Done()

	for num := range ch {
		result = append(result, num)
	}
	fmt.Printf("Printer %d: %v\n", id, result)
}

func main() {
	sendCh := make(chan int)
	evenCh := make(chan int)
	oddCh := make(chan int)
	done := make(chan bool)
	var wg sync.WaitGroup

	// Start sender processes
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go senderProcess(i, sendCh, &wg)
	}

	// Start receiver process
	wg.Add(1)
	go receiverProcess(sendCh, evenCh, oddCh, done, &wg)

	// Start printer processes
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go printerProcess(i+1, evenCh, []int{}, &wg)
	}

	for i := 0; i < 2; i++ {
		wg.Add(1)
		go printerProcess(i+3, oddCh, []int{}, &wg)
	}

	// Wait for the receiver to finish
	<-done

	// Wait for all goroutines to finish
	wg.Wait()
}
