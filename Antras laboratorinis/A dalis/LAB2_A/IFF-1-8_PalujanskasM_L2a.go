package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"time"
)

var (
	DataSize    = 30
	WorkerCount = DataSize / 4
)

type Cars struct {
	Cars []Car `json:"cars"`
}

type Car struct {
	Manufacture string  `json:"manufacture"`
	Year        int     `json:"year"`
	Engine      float32 `json:"engine"`
}

// Lyginimo funkcija
func (car *Car) Compare(other *Car) bool {
	if car.Year == other.Year {
		return car.Engine > other.Engine
	} else {
		return car.Year > other.Year
	}
}

type Result struct {
	Car         *Car
	ResultValue [32]byte // byte hash
}

func (cars *Cars) ReadJsonCars(fileName string) {
	file, err := os.OpenFile(fileName, os.O_RDONLY, 0600)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	byteValue, _ := ioutil.ReadAll(file)

	err = json.Unmarshal(byteValue, &cars)
	if err != nil {
		panic(err)
	}
}

// Main thread
func main() {

	t := time.Now()
	var cars Cars
	//1 nuskaitom nuomenis
	cars.ReadJsonCars("IFF-1-8_PalujanskasM_L2_dat_2.json")

	mainRequestChan := make(chan int) // Signalu kanalai
	workerRequestChan := make(chan int)

	mainToDataChan := make(chan Car) // Duomenu kanalai
	dataToWorkerChan := make(chan *Car)
	workerToResultChan := make(chan *Result)
	workerExitChan := make(chan int)
	resultToMainChan := make(chan []Result)
	resultCountChan := make(chan int)

	//duomenu masyvo pripildymas ir istustejimas
	mainControlChan := make(chan string)
	// 3.1 Papildoma logika vykdant main proceso veiksmus
	go func() {
		for {
			select {
			case command := <-mainControlChan:
				switch command {
				case "addData":
					// Pridėti duomenis į duomenų masyvą
					for i := 0; i < DataSize/2; i++ {
						// Galima pavyzdžiui generuoti atsitiktinius automobilius
						newCar := Car{
							Manufacture: fmt.Sprintf("Brand%d", i),
							Year:        2020,
							Engine:      1.5,
						}
						mainToDataChan <- newCar
					}

				case "clearData":
					// Ištuštiname duomenų masyvą
					for i := 0; i < DataSize/2; i++ {
						<-mainToDataChan
					}
				}

			case <-time.After(time.Second * 1):
				// Periodiškai vykdome kitus veiksmus pagrindiniam procesui
				// Šiuo atveju tikriname duomenų masyvo būklę ir atliekame veiksmus pagal poreikį.
				if len(mainToDataChan) == 0 {
					// Jei duomenų masyvas tuščias, galime pridėti duomenų.
					mainControlChan <- "addData"
				} else if len(mainToDataChan) >= DataSize/2 {
					// Jei duomenų masyvas pilnas, galime jį ištuštinti.
					mainControlChan <- "clearData"
				}
			}
		}
	}()

	// 2 Paleidžiame darbinius procesus
	for i := 0; i < WorkerCount; i++ {
		go WorkerProcess(dataToWorkerChan, workerToResultChan, workerRequestChan, workerExitChan)
	}

	// Paleidžiame duomenų ir rezultatų valdymo procesus
	go DataProcess(mainToDataChan, dataToWorkerChan, workerRequestChan, mainRequestChan)
	go ResultProcess(workerToResultChan, resultToMainChan, resultCountChan)

	// 3 Duomenų masyvą valdančiam procesui po vieną persiunčia visus nuskaitytus elementus iš failo
	for _, car := range cars.Cars {
		mainRequestChan <- 1
		mainToDataChan <- car
	}
	close(mainRequestChan)
	// Papildomas kodas, kad būtų galima ištestuoti addData
	//mainControlChan <- "addData"
	// Palaukiame, kol main thread užbaigs pridėjimą
	//time.Sleep(time.Second * 2)
	// Papildomas kodas, kad būtų galima ištestuoti clearData
	//mainControlChan <- "clearData"
	// Palaukiame, kol main thread užbaigs ištuštinimą
	//time.Sleep(time.Second * 2)

	for i := 0; i < WorkerCount; i++ {
		<-workerExitChan
	}
	fmt.Println("Visi workeriai baige darba")
	close(workerToResultChan) // Uždarome workerių į rezultatų kanalą

	// 4 iš rezultatų masyvą valdančio proceso gauname rezultatus
	results := <-resultToMainChan
	count := <-resultCountChan

	//5 rezultatų išvedimas į failą
	resultFile, err := os.OpenFile("IFF-1-8_PalujanskasM_L2_rez.txt", os.O_TRUNC|os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		panic(err)
	}
	defer resultFile.Close()

	resultFile.WriteString(fmt.Sprintf("            Sorted cars, that were built less than 10 years ago:\n"))
	resultFile.WriteString(fmt.Sprintf("Nr: %-5v |Manufacture: %-10v |Year: %-5v |Engine: %-5v |Hash: \n", "", "", "", ""))

	for i := 0; i < count; i++ {
		resultFile.WriteString(fmt.Sprintf("%-9v |%-23v |%-11v |%-13v |%x\n",
			i+1,
			results[i].Car.Manufacture,
			results[i].Car.Year,
			results[i].Car.Engine,
			results[i].ResultValue))
	}
	fmt.Println("Vykdymo laikas:", time.Since(t))

}

// WorkerProcess - vykdo skaičiavimus ir perduoda rezultatus result thread
func WorkerProcess(dataToWorkerChan <-chan *Car, workerToResultChan chan<- *Result, workerRequestChan, workerExitChan chan<- int) {

	for {
		workerRequestChan <- 1    // 1 Prašo data thread duomenu
		car := <-dataToWorkerChan // Gauna duomenis
		if car == nil {           // Data thread baige siusti duomenis ir kanalas yra uzdarytas
			break
		}

		//2 Skaičiuojamas operacijos rezultatas
		stringToHash := fmt.Sprintf("%v %v %v", car.Manufacture, car.Year, car.Engine)
		var hash [32]byte
		hash = sha256.Sum256([]byte(stringToHash)) // vyksta hashinimas
		for i := 0; i < 1000; i++ {
			hash = sha256.Sum256([]byte(fmt.Sprintf("%v%x", i, hash)))
		}

		//3 Jei tenkina, siunčia rezultatą į result thread
		if car.Year > 2012 {
			workerToResultChan <- &Result{car, hash}
		}
	}
	workerExitChan <- 0 // informuoja main thread, kad workeris baige darba
}

// DataProcess - gauna duomenis ir perduoda juos WorkerFunction'ui.
func DataProcess(mainToDataChan <-chan Car, dataToWorkerChan chan<- *Car, workerRequestChan, mainRequestChan chan int) {
	arraySize := DataSize / 2
	localArray := make([]Car, arraySize) //1
	count := 0
	isDone := false
	//2
	for !isDone || count > 0 {
		if count >= arraySize {
			// Jei duomenų masyvas pilnas, laukia žinutės iš šalinančio proceso
			<-workerRequestChan
			value := localArray[count-1]
			dataToWorkerChan <- &value
			count--
		} else if count <= 0 && !isDone {
			// Jei duomenų masyvas tuščias, laukia žinutės iš įterpiančio proceso
			<-mainRequestChan
			localArray[count] = <-mainToDataChan
			count++
		} else {
			select {
			case request := <-mainRequestChan:
				// Įterpimo proceso žinutė
				if request == 0 {
					isDone = true
					break
				}
				input := <-mainToDataChan
				localArray[count] = input
				count++
				break
			case <-workerRequestChan:
				// Šalinimo proceso žinutė
				value := localArray[count-1]
				dataToWorkerChan <- &value
				count--
				break
			}
		}
	}

	close(dataToWorkerChan)
	for i := 0; i < WorkerCount; i++ {
		// Laukia žinutės iš visų šalinančių procesų
		<-workerRequestChan
	}
	close(workerRequestChan)
}

// ResultProcess - tvarko ir rikiuoja rezultatus, galiausiai juos perduoda main thread
func ResultProcess(workerToResultChan <-chan *Result, resultToMainChan chan<- []Result, resultCountChan chan<- int) {
	// 1
	var localArray []Result

	for {
		//2
		result := <-workerToResultChan // Laukiame naujo rezultato iš Worker thread.
		if result == nil {             // Jei gauname nil, tai reiškia, kad Worker thread baigė darbą.
			break
		}

		addedInLoop := false

		// Tikriname, ar naujas rezultatas sutampa su jau esančiais rezultatais.
		for index, existingResult := range localArray {
			if existingResult.Car.Compare(result.Car) {
				localArray = append(localArray[:index+1], localArray[index:]...)
				localArray[index] = *result
				addedInLoop = true
				break
			}
		}

		if !addedInLoop {
			localArray = append(localArray, *result)
		}
	}

	fmt.Println("Rezultatu gija baige darba ir siuncia rezultatus i pagrindine gija")
	resultToMainChan <- localArray     // Siunčiame surūšiuotus rezultatus į main thread
	resultCountChan <- len(localArray) // Siunčiame rezultatų skaičių į main thread
}
