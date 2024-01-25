#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <string>

using namespace std;

const int MAX_NAME = 17;

class Car {
public:
    string manufacture;
    int year;
    float engine;
};

void GenerateRandomCar(Car& car, int index) {
    // Galimos automobilių markės ir variklių tipai
    string manufactures[] = { "Toyota", "Honda", "Ford", "Chevrolet", "Nissan", "Audi", "Volkswagen", "Mazda", "Hyundai", "Kia", "Lexus", "Infiniti", "Alfa-Romeo", "Jaguar", "Land-Rover", "Mercedes-Benz", "BMW", "Subaru", "Mitsubishi", "Porsche", "Ferrari", "Lamborghini", "McLaren", "Bugatti", "Aston-Martin" };
    int manufactureCount = sizeof(manufactures) / sizeof(manufactures[0]);

    // Sugeneruojame atsitiktinius duomenis
    car.manufacture = manufactures[rand() % manufactureCount] + to_string(index + 1); // Pridedame unikalų skaitinį
    car.year = rand() % 10 + 1999;  // Metai nuo 2014 iki 2023
    car.engine = static_cast<float>(rand() % 40) / 10.0 + 4.0;  // Variklio tūris nuo 1.0 iki 6.0
}

int main() {
    // Nustatome atsitiktinių skaičių generatoriaus sėklę
    srand(static_cast<unsigned>(time(nullptr)));

    const int totalCars = 250;
    Car cars[totalCars];

    // Generuojame automobilių duomenis
    for (int i = 0; i < totalCars; i++) {
        GenerateRandomCar(cars[i], i);
    }

    // Išvedame sugeneruotus duomenis į failą
    ofstream outFile("IFF-1-8_PalujanskasM_L3_dat_1.txt");

    for (int i = 0; i < totalCars; i++) {
        outFile << cars[i].manufacture << " " << cars[i].year << " " << cars[i].engine << "\n";
    }

    outFile.close();

    cout << "Sugeneruoti automobilių duomenys ir įrašyti į failą 'IFF-1-8_PalujanskasM_L3_dat_1.txt'." << endl;

    return 0;
}
