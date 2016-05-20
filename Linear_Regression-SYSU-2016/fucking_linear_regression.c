#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>

#define M 25000
#define MAX_J 385
#define BUF_SIZE 100000
#define LEARNING_RATE 0.003

double feature[M][MAX_J];
double reference[M];
double sita[MAX_J];

void read_feature() {
    int i, j, id;
    for (i = 0; i < M; i++)
        feature[i][0] = 1;
    FILE *train_data = fopen("./train.csv", "r");
    char *unuse_line = (char*)malloc(sizeof(char) * BUF_SIZE);
    fgets(unuse_line, BUF_SIZE, train_data);
    free(unuse_line);
    for (i = 0; i < M; i++) {
        fscanf(train_data, "%d,", &id);
        for (j = 1; j < MAX_J; j++)
            fscanf(train_data, "%lf,", &feature[i][j]);
        fscanf(train_data, "%lf\n", &reference[i]);
    }
    fclose(train_data);
}

void feature_scaling() {
    int i, j;
    double max, min, average;
    for (j = 1; j < MAX_J; j++) {
        max = min = average = feature[0][j];
        for (i = 1; i < M; i++) {
            if (feature[i][j] > max)
                max = feature[i][j];
            if (feature[i][j] < min)
                min = feature[i][j];
            average += feature[i][j] / (double)M;
        }
        for (i = 0; i < M; i++)
            feature[i][j] = (max - min < 1e-7) ? 0 : (feature[i][j] - average) / (max - min);
    }
}

double cost_func() {
    double cost = 0, h;
    int i, j;
    for (i = 0; i < M; i++) {
        h = 0;
        for (j = 0; j < MAX_J; j++)
            h += sita[j] * feature[i][j];
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
                h += sita[k] * feature[i][k];
            factor = h - reference[i];
            for (j = 0; j < MAX_J; j++) {
                sum = factor * feature[i][j];
                sita[j] -= LEARNING_RATE * (1 / (double)M) * sum;
            }
        }
    }
}

void predict_with_training_factor() {
    int i, j, id;
    double predict_value;
    FILE *test_data = fopen("./test.csv", "r");
    char *unuse_line = (char*)malloc(sizeof(char) * BUF_SIZE);
    fgets(unuse_line, BUF_SIZE, test_data);
    free(unuse_line);
    FILE *predict_data = fopen("./predict.csv", "w+");
    fprintf(predict_data, "Id,reference\n");
    for (i = 0; i < M; i++) {
        fscanf(test_data, "%d,", &id);
        for (j = 1; j < MAX_J; j++)
            fscanf(test_data, "%lf,", &feature[i][j]);
    }
    feature_scaling();
    for (i = 0; i < M; i++) {
        predict_value = 0;
        for (j = 0; j < MAX_J; j++)
            predict_value += sita[j] * feature[i][j];
        fprintf(predict_data, "%d,%lf\n", i, predict_value);
    }
    fclose(test_data);
    fclose(predict_data);
}

int main() {
    read_feature();
    feature_scaling();
    train();
    predict_with_training_factor();
    return 0;
}

