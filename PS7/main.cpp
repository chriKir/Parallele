//
// Created by Patrick Lanzinger on 28.11.16.
//
#include <iostream>
#include "include/ClWrapper.h"
#include "Search.h"




int main(void) {
    std::cout << "Start searching...\n";
    Search search;
    search.execute(1000);
    std::cout << "Finished searching...\n";
}



