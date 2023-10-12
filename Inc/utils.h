#ifndef __UTILS_H__
#define __UTILS_H__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cmatrix.h"

// 使用链式结构存储网络参数
// 每一层可以存储参数矩阵 W、b
// 多层Layer就可以构建一个Model
// 一个model就是一个由多个Layer节点构建的链表
// 只需要前向传播则只需要单链，需要反向传播则需要改为双链
typedef struct Layer{
    struct Matrix* W;
    struct Matrix* b;
    struct Layer* next;
}Layer;

// 函数声明
double relu(double x);
Matrix* relu_matrix(Matrix* matrix);
double sigmoid(double x);
Matrix* sigmoid_matrix(Matrix* matrix);
Matrix* forward(Matrix* X, Matrix* W, Matrix* b, int ctrl); // 节点运算
Layer* ModelCreate(double*** w, double*** b); // 创建模型
int GetLayerdims(Layer* model); // 获取网络层数
Layer* ModelPrint(Layer* model); // 打印模型参数
Matrix* predict(Matrix* X, Layer* model); // 预测函数

double relu(double x){return (x>0)?x:0;}
// 矩阵relu操作
Matrix* relu_matrix(Matrix* matrix){
    int rows=matrix->rows, cols=matrix->cols;
    MallocMatrix(result, rows, cols);
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            result->mat[i*cols+j] = relu(matrix->mat[i*cols+j]);
        }
    }
    return result;
}

double sigmoid(double x){return 1.0/(1.0+exp(-x));}
// 矩阵sigmoid操作
Matrix* sigmoid_matrix(Matrix* matrix){
    int rows=matrix->rows, cols=matrix->cols;
    MallocMatrix(result, rows, cols);
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            result->mat[i*cols+j] = sigmoid(matrix->mat[i*cols+j]);
        }
    }
    return result;
}

// 感知机节点矩阵运算函数
// ctrl{0: relu, 1: sigmoid}
Matrix* forward(Matrix* X, Matrix* W, Matrix* b, int ctrl){
    Matrix* result = MatrixMul(X,W);
    result = MatrixAdd(result,b);
    result = (!ctrl)?relu_matrix(result):sigmoid_matrix(result);
    return result;
}

Layer* ModelCreate(double*** w, double*** b){
    // 通过传递的参数获取网络层数，每层网络节点数, 输入数据的维度并将参数保存到模型参数矩阵中
    int layerdim=0; // 网络层数
    while (w[layerdim]!=NULL){
        layerdim++;
    }
    int neurondim[layerdim+1];
    int inputshape=0; // 输入数据的维度
    while (w[0][inputshape]!=NULL){
        inputshape++;
    }
    neurondim[0]=inputshape;
    for (int i = 0; i < layerdim; i++){
        for (int j=0;w[i][j]!=NULL;j++){
            int n=0; // 每层网络节点数
            while (w[i][j][n]!=9999){
                n++;
            }
            neurondim[i+1]=n;
        }
    }
    
    // 创建模型
    Layer* model=(Layer*)malloc(sizeof(Layer)); // 创建网络层链表的头节点，也就是模型(model)节点
    model->next=NULL;
    Layer* r=model; // 创建尾指针，始终指向链表的尾节点，一开始指向头节点
    for (int i = 0; i < layerdim; i++){
        Layer* L=(Layer*)malloc(sizeof(Layer)); // 创建新的层节点
        int rows=neurondim[i], cols=neurondim[i+1]; // 获取该层中参数的维度
        MallocMatrixNotNew(L->W, rows, cols); // 为参数矩阵申请空间
        MallocMatrixNotNew(L->b, 1, cols);
        for (int j = 0; j < rows; j++){ // 权重参数W赋值
            for (int k = 0; k < cols; k++){
                L->W->mat[j*cols+k]=w[i][j][k];
            }
        }
        for (int k = 0; k < cols; k++){ // 偏置参数b赋值
            L->b->mat[k]=b[i][0][k];
        }
        r->next=L;r=L; // 将网络层节点插入模型节点
    }
    r->next=NULL;
    return model;
}

// 获取网络层数
int GetLayerdims(Layer* model){
    int layerdim=1; // 网络层数
    Layer* L=model->next;
    Layer* r=L;
    while (r->next!=NULL){
        layerdim++;
        r=r->next;
    }
    return layerdim;
}

Layer* ModelPrint(Layer* model){
    int layerdim=GetLayerdims(model);
    Layer* L=model->next;
    Layer* r=L;
    int i = 0;
    for (i = 0; i < layerdim; i++){
        int l=i+1;
        int W_rows=r->W->rows, W_cols=r->W->cols;
        int b_rows=r->b->rows, b_cols=r->b->cols;
        printf("------------ layer%d ------------\n",l);
        printf("W%d shape:[%d, %d] ", l, W_rows, W_cols);
        printf("b%d shape:[%d, %d]\n", l, b_rows, b_cols);
        // printf("------------ W%d ------------\n",l);
        // PrintMatrix(r->W);
        // printf("------------ b%d ------------\n",l);
        // PrintMatrix(r->b);
        // printf("---------------------------------\n");
        r=r->next;
    }
    printf("--------------------------------\n");
    printf("layerdim: %d\n", layerdim);
}
Matrix* predict(Matrix* X, Layer* model){
    Layer* L=model->next;
    int inputshape=L->W->rows;
    // printf("model input shape: [nan, %d]\n", inputshape);
    if(inputshape!=X->cols){
        printf("error: Input shape should be [1, %d], not [1, %d]\n", inputshape, X->cols);
        return NULL;
    }
    Layer* r=L;
    int layerdim=GetLayerdims(model);
    for (int i = 0; i < layerdim; i++){
        X = (i==layerdim-1)?forward(X,r->W,r->b,1):forward(X,r->W,r->b,0);
        r=r->next;
    }
    return X;
}

#endif
