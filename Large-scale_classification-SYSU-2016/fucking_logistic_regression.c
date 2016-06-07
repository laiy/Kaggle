#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>

#define M 300000
#define MAX_J 3183
#define LEARNING_RATE 0.0005

bool feature[M][MAX_J];
bool reference[M];
double sita[MAX_J];
short feature_map[12000];

void pre_handle_data() {
    memset(feature_map, -1, sizeof(feature_map));
    int feature_count = 1, id, feature;
    char c;
    FILE *test_data = fopen("./test.txt", "r");
    while (fscanf(test_data, "%d", &id) != EOF) {
        while (fscanf(test_data, "%d:1", &feature)) {
            if (feature_map[feature] == -1)
                feature_map[feature] = feature_count++;
            if ((c = fgetc(test_data)) == '\n')
                break;
        }
    }
    fclose(test_data);
    printf("max feature count: %d\n", --feature_count);
}

void read_feature() {
    int i, r, f;
    char c;
    memset(feature, false, sizeof(feature));
    for (i = 0; i < M; i++)
        feature[i][0] = 1;
    FILE *train_data = fopen("./train.txt", "r");
    for (i = 0; i < M; i++) {
        fscanf(train_data, "%d", &r);
        reference[i] = r;
        while (fscanf(train_data, "%d:1", &f)) {
            if (feature_map[f] != -1)
                feature[i][feature_map[f]] = 1;
            if ((c = fgetc(train_data)) == '\n')
                break;
        }
    }
    fclose(train_data);
}

inline double sigmoid(double f) {
    return 1 / (1 + exp(-f));
}

double cost_func() {
    int i, j;
    double cost = 0, h, sum;
    for (i = 0; i < M; i++) {
        sum = 0;
        for (j = 0; j < MAX_J; j++)
            sum += sita[j] * feature[i][j];
        h = sigmoid(sum);
        cost += -reference[i] * log(h) - (1 - reference[i]) * log(1 - h);
    }
    return cost;
}

void train() {
    memset(sita, 0, sizeof(sita));
    double pre_cost = cost_func() + 1;
    double cost, h, factor;
    int i, j, k;
    while ((cost = cost_func()) < pre_cost) {
        pre_cost = cost;
        printf("cost: %lf\n", cost);
        for (i = 0; i < M; i++) {
            h = 0;
            for (k = 0; k < MAX_J; k++)
                h += sita[k] * feature[i][k];
            factor = sigmoid(h) - reference[i];
            for (j = 0; j < MAX_J; j++)
                sita[j] -= LEARNING_RATE * factor * feature[i][j];
        }
    }
}

bool predict_feature[MAX_J];

void predict_with_training_factor() {
    int j, id, f;
    double predict_value;
    char c;
    FILE *test_data = fopen("./test.txt", "r");
    FILE *predict_data = fopen("./predict.csv", "w+");
    fprintf(predict_data, "id,label\n");
    while (fscanf(test_data, "%d", &id) != EOF) {
        memset(predict_feature, 0, sizeof(predict_feature));
        predict_feature[0] = 1;
        while (fscanf(test_data, "%d:1", &f)) {
            predict_feature[feature_map[f]] = 1;
            if ((c = fgetc(test_data)) == '\n')
                break;
        }
        predict_value = 0;
        for (j = 0; j < MAX_J; j++)
            predict_value += sita[j] * predict_feature[j];
        fprintf(predict_data, "%d,%d\n", id, int(round(sigmoid(predict_value))));
    }
    fclose(test_data);
    fclose(predict_data);
}

static void handler(int signo) {
    printf("signal %d handling...\n", signo);
    predict_with_training_factor();
    printf("done.\n");
    exit(0);
}

int main() {
    pre_handle_data();
    read_feature();
    signal(SIGINT, handler);
    train();
    predict_with_training_factor();
    return 0;
}

