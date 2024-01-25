#include <iostream>                // Įtraukiame C++ įvesties/išvesties funkcionalumą
#include <fstream>                // Įtraukiame failų įvesties/išvesties funkcionalumą
#include <thread>                // Įtraukiame gijų valdymo funkcionalumą
#include <string>                // Įtraukiame eilučių manipuliavimo funkcionalumą
#include <condition_variable>    // Įtraukiame sąlyginės sinchronizacijos funkcionalumą
#include <iomanip>                // Įtraukiame formatavimo funkcionalumą
#include <nlohmann/json.hpp>    // Įtraukiame JSON duomenų apdorojimo funkcionalumą iš nlohmann/json bibliotekos
#include <vector>                // Įtraukiame vektorių, kuris naudojamas laikyti gijas

using namespace std;
using json = nlohmann::json;

// Struktūra, kuri saugo duomenis apie automobilius
struct cars {
    string manufacture; // Automobilio gamintojas
    int year;           // Metai, kai buvo pagamintas automobilis
    double engine;      // Variklio tūris
    string hashed;      // Maiša, skaičiuojama pagal gamintoją, metus ir variklio tūrį
};

// Maišos funkcija, skirta apskaičiuoti hash reikšmę iš string'o
//source:https://www.youtube.com/watch?v=iN42o08Xwrk
unsigned int SHF(string input)
{
    unsigned int Init = 1254545142;
    unsigned int Magic = 5454545;
    unsigned int Hash;
    for (int i = 0; i < input.length(); i++)
    {
        Hash = Hash ^ (input[i]);
        Hash = Hash * Magic;
    }
    return Hash;
}

// Monitoriaus klasė, skirta saugoti duomenims ir apsaugoti nuo gijų konkurencijos
class Monitor
{
    cars monitorCars[15];            // Masyvas, skirtas saugoti duomenims
    std::condition_variable cv;      // Sąlyginė sąsaja gijoms sinchronizuoti
    int count = 0;                   // Kiekis saugomų duomenų
public:
    mutex mtx2;                      // Mutex užrakinimui
    bool isGoingOn = true;           // Ar darbininkės gijos tęs darbą
    bool shouldReturnEmpty = false;  // Ar monitorius turėtų grąžinti tuščią elementą

    void notify()
    {
        cv.notify_all(); // Praneša visoms gijoms apie vykusį įvykį
    }

    void push(cars newOne)
    {
        std::unique_lock<std::mutex> lck(mtx2); // Užrakina kritinę sekciją su mutex
        while (count == 15) // Jeigu pasiektas masyvo limitas, gija užmiega
        {
            cv.wait(lck); // Atlaisvina mutex ir užmigsta
        }
        monitorCars[count] = newOne; // Įdeda naują elementą į masyvą
        count++;
        cv.notify_all(); // Praneša kitoms gijoms, kad gali tęsti darbą
    }

    cars pop() // Ištrina ir grąžina automobilį iš monitoriaus masyvo
    {
        cars carR;
        std::unique_lock<std::mutex>lck(mtx2); // Užrakina kritinę sekciją su mutex
        while (count == 0) // Jeigu masyvas tuščias, gija užmiega
        {
            if (shouldReturnEmpty)
            {
                carR.year = 0000;
                return carR;
            }
            cv.wait(lck); // Atlaisvina mutex ir užmigsta
        }
        carR = monitorCars[count - 1]; // Paima paskutinį elementą iš masyvo
        count--;
        cv.notify_all(); // Praneša kitoms gijoms, kad gali tęsti darbą
        return carR;
    }

    int getCount()
    {
        return count; // Grąžina saugomų duomenų kiekį
    }
};

Monitor monitor;

// Monitoriaus klasė rezultatams saugoti
class Monitor_results
{
    std::condition_variable cv; // Sąlyginė sąsaja gijoms sinchronizuoti

public:
    cars monitorCars[40];      // Masyvas, skirtas saugoti rezultatams
    int count;                 // Kiekis saugomų rezultatų
    mutex mtx2;                // Mutex užrakinimui
    bool isGoingOn = true;     // Ar darbininkės gijos tęs darbą

    void pushSorted(cars newOne)
    {
        std::unique_lock<std::mutex> lck(mtx2); // Užrakina kritinę sekciją su mutex

        if (newOne.year < 2012) // Filtruoja pagal metus
        {
            return;
        }

        int i;
        for (i = count - 1; (i >= 0 && monitorCars[i].year > newOne.year); i--)
        {
            monitorCars[i + 1] = monitorCars[i]; // Perstumia automobilius pagal metus, kad įterptų naują
        }

        // Naują automobilį įterpiame pagal metus
        monitorCars[i + 1] = newOne;      

        count++;
    }

};


Monitor_results monitor2;

void execute(int threadCount)
{
    // Tikrina, ar vis dar yra duomenų, ar darbininkės gijos turi tęsti darbą
    while (monitor.isGoingOn || monitor.getCount() > 0)
    {
        cout << to_string(threadCount) + " Thread - EXECUTE! Dar neapdorota - " + to_string(monitor.getCount()) + " cars!\n";
        cars manufacture = monitor.pop(); // Paima automobilį iš duomenų monitoriaus
        if (manufacture.year == 0000) // Papildomas patikrinimas, kad nebūtų grąžinamas pradinis tuščias automobilio elementas
            break;
        manufacture.hashed = to_string(SHF(manufacture.manufacture + to_string(manufacture.year) + to_string(manufacture.engine))); // Apskaičiuoja hash reikšmę
        cout << manufacture.manufacture + " kuris buvo pagamintas " + to_string(manufacture.year) + ", variklis - " + to_string(manufacture.engine) + " , hash kodas = " + manufacture.hashed + "\n";
        monitor2.pushSorted(manufacture); // Įdeda automobilį į rezultatų monitorių su filtravimu
    }
}

int main() {
    int threadCount = 4;
    //ifstream t("IFF-1-8_PalujanskasM_L1_dat_1.json"); // Atidaro JSON failą skaitymui
    ifstream t("IFF-1-8_PalujanskasM_L1_dat_2.json");
    //ifstream t("IFF-1-8_PalujanskasM_L1_dat_3.json");

    string jsonFile((istreambuf_iterator<char>(t)), istreambuf_iterator<char>()); // Perskaito .json failą į string

    auto j = json::parse(jsonFile); // Transformuojamas JSON failo turinys

    cars mainCars[40]; // Masyvas saugoti pagrindinius duomenis
    int mainCarsCount = 0;

    // Krauna JSON duomenis į struct cars
    for (size_t i = 0; i < j["cars"].size(); i++)
    {
        cars tempCar = {
            j["cars"][i]["manufacture"],
            j["cars"][i]["year"],
            j["cars"][i]["engine"]
        };

        mainCars[mainCarsCount++] = tempCar; // Įdeda automobilį į pagrindinį masyvą
    }

    // Sukuria ir paleidžia darbininkės gijas
    std::vector<std::thread> threads(threadCount);

    for (int i = 0; i < threadCount; ++i)
        threads[i] = std::thread(execute, i + 1);

    // Pagrindinė gija pradeda duomenų įvedimą
    while (mainCarsCount > 0)
    {
        monitor.push(mainCars[mainCarsCount - 1]); // Įdeda automobilį į duomenų monitorių
        mainCarsCount--;
        cout << "Main Thread -  PUSH. Left: " + to_string(mainCarsCount) + "\n";
    }

    // Informuoja monitoriaus klasę, kad daugiau PUSH nebus daromas
    monitor.isGoingOn = false;
    // Užtikrina, kad neliks jokio POP metodo laukiančio naujo įvesties
    monitor.shouldReturnEmpty = true;
    monitor.notify(); // Praneša visoms gijoms apie vykusį įvykį

    // Laukia, kol visos darbininkės gijos baigs darbą
    for (auto& th : threads) th.join();

    // Išveda rezultatus į failą
    ofstream myfile;
    myfile.open("IFF-1-8_PalujanskasM_L1_rez.txt");

    myfile << "-------------------------------------------------------------------" << endl;
    myfile << "| Sorted cars, that were built less than 10 years ago:" << endl;
    myfile << "-------------------------------------------------------------------" << endl;
    for (int i = 0; i < monitor2.count; i++)
    {
        string resultOut = monitor2.monitorCars[i].manufacture + ", year of production: " + to_string(monitor2.monitorCars[i].year) + ", engine: " + std::to_string(monitor2.monitorCars[i].engine) + ", hash code = " + monitor2.monitorCars[i].hashed + "\n";
        cout << resultOut;
        myfile << "| " << left << setw(20) << monitor2.monitorCars[i].manufacture << left << setw(20) << fixed << std::setprecision(1) << monitor2.monitorCars[i].engine << left << setw(20) <<
            to_string(monitor2.monitorCars[i].year) << monitor2.monitorCars[i].hashed + "\n";
    }
    myfile << "-------------------------------------------------------------------" << endl;
    myfile.close();

    return 0;
}


