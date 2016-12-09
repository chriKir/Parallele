//
// Created by Patrick Lanzinger on 09.12.16.
//

#include <include/people.h>

#define AGE 120

#ifndef PARALLELE_SERIAL_COUNT_SORT_H
#define PARALLELE_SERIAL_COUNT_SORT_H

class serial_count_sort {
private:
    person_t *people;
    int number_of_people = 0;

public:
    serial_count_sort(person_t *people, int number_of_people) {
        this->people = people;
        this->number_of_people = number_of_people;
    }

    person_t *sort() {
        person_t output[number_of_people];

        int count[AGE];
        memset(count, 0, sizeof(count));

        for (int i = 0; i < number_of_people; i++) {
            ++count[people[i].age];
        }

        for (int i = 1; i < AGE; i++) {
            count[i] += count[i - 1];
        }

        for (int i = 0; i < number_of_people; i++) {
            output[count[this->people[i].age] - 1] = people[i];
            --count[this->people[i].age];
        }

        for (int i = 0; i < number_of_people; i++) {
            this->people[i] = output[i];
        }

        return this->people;

    }
};

#endif //PARALLELE_SERIAL_COUNT_SORT_H
