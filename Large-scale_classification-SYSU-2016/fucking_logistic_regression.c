#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#define M 40000
#define MAX_J 3183
#define LEARNING_RATE 0.0007
#define MAX_BUF 10000
#define MAX_TRAINING_TIME 1000
#define PROGRESS_NUM 19
#define TEST_SIZE 220245

bool feature[M][MAX_J];
bool reference[M];
double sita[MAX_J];
short feature_map[12000];
int piece;

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
    printf("max feature count: %d.\n", --feature_count);
}

void cut_training_data() {
    FILE *train_data = fopen("./train.txt", "r");
    int pieces = PROGRESS_NUM, count;
    const char *pre_path = "./train";
    char path[20], readline[MAX_BUF];
    while (pieces--) {
        count = M;
        snprintf(path, sizeof(path), "%s%d.txt", pre_path, pieces);
        FILE *part_data = fopen(path, "w+");
        while (count--) {
            fgets(readline, MAX_BUF, train_data);
            fputs(readline, part_data);
        }
        fclose(part_data);
    }
    fclose(train_data);
    printf("data cut.\n");
}

void read_feature() {
    int i, r, f;
    char c;
    memset(feature, false, sizeof(feature));
    for (i = 0; i < M; i++)
        feature[i][0] = 1;
    char path[20];
    snprintf(path, sizeof(path), "./train%d.txt", piece);
    FILE *train_data = fopen(path, "r");
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
    printf("pid: %d, train data read: %s\n", getpid(), path);
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
    int i, j, k, count = 0;
    while ((cost = cost_func()) < pre_cost && count++ < MAX_TRAINING_TIME) {
        pre_cost = cost;
        printf("pid: %d, cost: %lf\n", getpid(), cost);
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
    char path[20];
    FILE *test_data = fopen("./test.txt", "r");
    snprintf(path, sizeof(path), "./predict%d.csv", piece);
    FILE *predict_data = fopen(path, "w+");
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
    printf("pid: %d, predicted: %s\n", getpid(), path);
}

static void handler(int signo) {
    printf("pid: %d, signal %d handling...\n", getpid(), signo);
    predict_with_training_factor();
    printf("pid: %d, done.\n", getpid());
    exit(0);
}

void vote_for_final_prediction() {
    char path[20];
    int id, prediction;
    short votes[TEST_SIZE];
    memset(votes, 0, sizeof(votes));
    for (int i = 0; i < PROGRESS_NUM; i++) {
        snprintf(path, sizeof(path), "./predict%d.csv", piece);
        FILE *predict_data = fopen(path, "r");
        while (fscanf(predict_data, "%d,%d\n", &id, &prediction) != EOF)
            votes[id] += prediction;
        fclose(predict_data);
    }
    FILE *final_predict_data = fopen("./predict.csv", "w+");
    fprintf(final_predict_data, "id,label\n");
    for (int i = 0; i < TEST_SIZE; i++) {
        fprintf(final_predict_data, "%d,%d\n", i, votes[i] > PROGRESS_NUM / 2 ? 1 : 0);
    }
    fclose(final_predict_data);
    printf("vote over, result in ./predict.csv\n");
}

int main() {
    pre_handle_data();
    cut_training_data();
    pid_t pid;
    for (int i = 0; i < PROGRESS_NUM; i++) {
        if ((pid = fork()) == 0) {
            piece = i;
            break;
        } else if (i == PROGRESS_NUM - 1) {
            signal(SIGINT, SIG_IGN);
            int status;
            for (i = 0; i < PROGRESS_NUM; i++) {
                pid = wait(&status);
                printf("the return code is %d.\n", WEXITSTATUS(status));
            }
            printf("all child process done, now voting the final result due to all results predicted by child process.\n");
            vote_for_final_prediction();
            exit(0);
        }
    }
    printf("pid: %d, piece: %d\n", getpid(), piece);
    read_feature();
    signal(SIGINT, handler);
    train();
    predict_with_training_factor();
    return 0;
}

