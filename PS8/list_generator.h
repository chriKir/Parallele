//
// Created by Patrick Lanzinger on 09.12.16.
//
#include <include/people.h>

#ifndef PARALLELE_LIST_GENERATOR_H
#define PARALLELE_LIST_GENERATOR_H

class list_generator {
private:
    unsigned int number_of_people;
    person_t *people;


public:

    //constructor
    list_generator(const unsigned int number_of_people, const unsigned int ranom_init) {
        this->number_of_people = number_of_people;

        people = (person_t*)malloc(sizeof(person_t) * number_of_people);
        srand(ranom_init);
    }

    //deconstructor
    ~list_generator() {
        free(people);
    }

    void generate_list() {
        for (int i = 0; i < number_of_people; i++) {
            gen_name(people[i].name);

            //max age is 119
            people[i].age = rand() % 120;
        }
    }

    void print_list() {
        if (this->people == NULL) {
            std::cout << "list has not been generated\n";
            return;
        }

        for (int i = 0; i < number_of_people; i++) {
            std::cout << people[i].age << "\t|| " << people[i].name << "\n";
        }
    }

};

#endif //PARALLELE_LIST_GENERATOR_H
