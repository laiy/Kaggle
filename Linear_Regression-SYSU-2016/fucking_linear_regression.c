#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>

#define M 25000
#define MAX_J 385
#define BUF_SIZE 100000
#define LEARNING_RATE 0.003

double training_data[M][MAX_J];
double reference[M];
double sita[MAX_J];

void read_training_data() {
    int i, j;
    double id;
    for (i = 0; i < M; i++)
        training_data[i][0] = 1;
    FILE *train_data = fopen("./train.csv", "r");
    char *unuse_line = (char*)malloc(sizeof(char) * BUF_SIZE);
    fgets(unuse_line, BUF_SIZE, train_data);
    free(unuse_line);
    for (i = 0; i < M; i++) {
        fscanf(train_data, "%lf,", &id);
        for (j = 1; j < MAX_J; j++)
            fscanf(train_data, "%lf,", &training_data[i][j]);
        fscanf(train_data, "%lf\n", &reference[i]);
    }
}

void feature_scaling() {
    int i, j;
    double max, min, average;
    for (j = 1; j < MAX_J; j++) {
        max = min = average = training_data[0][j];
        for (i = 1; i < M; i++) {
            if (training_data[i][j] > max)
                max = training_data[i][j];
            if (training_data[i][j] < min)
                min = training_data[i][j];
            average += training_data[i][j] / (double)M;
        }
        for (i = 0; i < M; i++)
            training_data[i][j] = (max - min < 1e-7) ? 0 : (training_data[i][j] - average) / (max - min);
    }
}

double cost_func() {
    double cost = 0, h;
    int i, j;
    for (i = 0; i < M; i++) {
        h = 0;
        for (j = 0; j < MAX_J; j++)
            h += sita[j] * training_data[i][j];
        cost += pow(h - reference[i], 2);
    }
    return cost;
}

void train() {
    memset(sita, 0, sizeof(sita));
    double pre_cost = cost_func() + 1;
    double cost, sum, h, factor;
    int i, j, k;
    while ((cost = cost_func()) < pre_cost) {
        pre_cost = cost;
        printf("cost: %lf\n", cost);
        for (i = 0; i < M; i++) {
            h = 0;
            for (k = 0; k < MAX_J; k++)
                h += sita[k] * training_data[i][k];
            factor = h - reference[i];
            for (j = 0; j < MAX_J; j++) {
                sum = factor * training_data[i][j];
                sita[j] -= LEARNING_RATE * (1 / (double)M) * sum;
            }
        }
    }
}

void predict_with_test_data() {
}

int main() {
    read_training_data();
    feature_scaling();
    train();
    predict_with_test_data();
    return 0;
}

