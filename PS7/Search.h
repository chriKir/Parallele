

#ifndef PARALLELE_SEARCH_H
#define PARALLELE_SEARCH_H


class Search {
private:
    void fill_array_with_random_numbers(double* array, int size);
public:
    Search();
    void execute(int iterations, int size);

};


#endif //PARALLELE_SEARCH_H
