//
// Created by Patrick Lanzinger on 09.12.16.
//
#include <iostream>
#include "list_generator.h"


int main(int argc, const char *argv[]) {

    if (argc != 3) {
        std::cout << "wrong!\n";
        exit(1);
    }

    const unsigned int number_of_people = atoi(argv[1]);
    const unsigned int random_init = atoi(argv[2]);

    list_generator gen(number_of_people, random_init);
    person_t *people = gen.generate_list();
    list_generator::print_list(people, number_of_people);

    free(people);


    return EXIT_SUCCESS;
}

