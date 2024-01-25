#include <iostream>
#include <mpi.h>
#include <random>


using namespace std;
using namespace MPI;

const int FIRST_SENDER = 0;
const int SECOND_SENDER = 1;
const int RECEIVER = 2;
const int EVEN = 3;
const int ODD = 4;

const int TAG_RECEIVE = 0;
const int TAG_END_SEND = 1;
const int TAG_PRINTER_EVEN = 2;
const int TAG_PRINTER_ODD = 3;

void Printer_Odd() {
    const auto MAX_SIZE = 20;
    int array[MAX_SIZE];
    int i = 0;
    Status status;
    bool gather = true;
    while (gather) {
        COMM_WORLD.Probe(RECEIVER, ANY_TAG, status);
        if (status.Get_tag() != TAG_END_SEND) {
            int message;
            COMM_WORLD.Recv(&message, 1, INT, RECEIVER, TAG_PRINTER_ODD);
            array[i] = message;
            i++;
        }
        else {
            COMM_WORLD.Recv(NULL, 0, INT, RECEIVER, TAG_END_SEND);
            gather = false;
        }
    }
    for (size_t j = 0; j < i; j++)
    {
        cout << "ODD Printer - " << array[j] << endl;
    }


}
void Printer_Even() {
    const auto MAX_SIZE = 20;
    int array[MAX_SIZE];
    int i = 0;
    Status status;
    bool gather = true;
    while (gather) {
        COMM_WORLD.Probe(RECEIVER, ANY_TAG, status);
        if (status.Get_tag() != TAG_END_SEND) {
            int message;
            COMM_WORLD.Recv(&message, 1, INT, RECEIVER, TAG_PRINTER_EVEN);
            array[i] = message;
            i++;
        }
        else {
            COMM_WORLD.Recv(NULL, 0, INT, RECEIVER, TAG_END_SEND);
            gather = false;
        }
    }
    for (size_t j = 0; j < i; j++)
    {
        cout << "EVEN Printer - " << array[j] << endl;
    }
}


void Receiver() {
    int counter = 0;
    int continue_task = 0;
    Status status;
    while (counter != 20) {
        COMM_WORLD.Probe(ANY_SOURCE, TAG_RECEIVE, status);
        int message;
        COMM_WORLD.Recv(&message, 1, INT, status.Get_source(), TAG_RECEIVE);
        COMM_WORLD.Send(&continue_task, 1, INT, status.Get_source(), TAG_END_SEND);
        if (message % 2 == 0) {
            //cout << "gavau " << message << " is " << status.Get_source() << endl;
            COMM_WORLD.Send(&message, 1, INT, EVEN, TAG_PRINTER_EVEN);
        }
        else {
            //cout << "gavau " << message << " is " << status.Get_source() << endl;
            COMM_WORLD.Send(&message, 1, INT, ODD, TAG_PRINTER_ODD);
        }
        counter++;
    }
    continue_task = 1;
    COMM_WORLD.Send(&continue_task, 1, INT, FIRST_SENDER, TAG_END_SEND);
    COMM_WORLD.Send(&continue_task, 1, INT, SECOND_SENDER, TAG_END_SEND);
    COMM_WORLD.Send(NULL, 0, INT, EVEN, TAG_END_SEND);
    COMM_WORLD.Send(NULL, 0, INT, ODD, TAG_END_SEND);
}






int main() {
    Init();
    auto totalProcesses = COMM_WORLD.Get_size();
    auto rank = COMM_WORLD.Get_rank();
    if (rank == 0) {
        bool send = true;
        Status status;
        int i = 0;
        while (send) {
            COMM_WORLD.Send(&i, 1, INT, RECEIVER, TAG_RECEIVE);
            i++;
            COMM_WORLD.Probe(RECEIVER, TAG_END_SEND, status);
            int receiver_message;
            COMM_WORLD.Recv(&receiver_message, 1, INT, RECEIVER, TAG_END_SEND);
            if (receiver_message == 1) {
                send = false;
            }
        }
        //cout << "pirmas sender baige" << endl;
    }
    else if (rank == 1) {
        bool send = true;
        Status status;
        int i = 11;
        while (send) {
            COMM_WORLD.Send(&i, 1, INT, RECEIVER, TAG_RECEIVE);
            i++;
            COMM_WORLD.Probe(RECEIVER, TAG_END_SEND, status);
            int receiver_message;
            COMM_WORLD.Recv(&receiver_message, 1, INT, RECEIVER, TAG_END_SEND);
            if (receiver_message == 1) {
                send = false;
            }
        }
        //cout << "antras sender baige" << endl;
    }
    else if (rank == 2) {
        Receiver();
    }
    else if (rank == 3) {
        Printer_Even();
    }
    else if (rank == 4) {
        Printer_Odd();
    }
    Finalize();
    return 0;
}
