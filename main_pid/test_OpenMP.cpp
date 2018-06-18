#include <iostream>
#include <thread>

int main(int argc, char ** argv)
{
    unsigned numCPU = std::thread::hardware_concurrency();
    std::cout << numCPU << std::endl;
    while(1);
}
