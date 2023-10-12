/* C语言矩阵运算库 */
/* 2023/10/07 by lucklu */
#ifndef __CMATRIX_H__
#define __CMATRIX_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 获取二维数组行数和列数
#define GET_MATRIX_SIZE(matrix,r,c) \
    int r=sizeof(matrix) / sizeof(matrix[0]);\
    int c=sizeof(matrix[0]) / sizeof(matrix[0][0])
// 为一维指针矩阵分配内存空间
#define MallocMat(mat, r, c)\
    mat=(double*)malloc(sizeof(double)*r*c)
// 新建矩阵并分配内存空间
#define MallocMatrix(m, r, c)\
    Matrix* m=(Matrix*)malloc(sizeof(Matrix));\
    m->rows=r;m->cols=c;\
    MallocMat(m->mat, r, c)
// 为已经创建了的矩阵分配内存空间
#define MallocMatrixNotNew(m, r, c)\
    m=(Matrix*)malloc(sizeof(Matrix));\
    m->rows=r;m->cols=c;\
    MallocMat(m->mat, r, c)
// 新建矩阵并初始化赋值
#define NewMatrix(m, r, c, matrix)\
    MallocMatrix(m, r, c);\
    InitMatrix(m,(double*)matrix)

typedef struct Matrix{
    int rows, cols;
    double* mat;
} Matrix;

// 函数声明
void InitMatrix(Matrix* arr, double* m); // 矩阵初始化赋值
void PrintMatrix(Matrix* matrix); // 打印矩阵
Matrix* copyMatrix(Matrix* matrix); // 复制矩阵
Matrix* TransMatrix(Matrix* matrix); // 转置矩阵
Matrix* MatrixAdd(Matrix* b_matrix, Matrix* c_matrix); // 矩阵加法(含广播机制)
Matrix* ScalarAdd(Matrix* matrix, double scalar); // 矩阵与常量相加
Matrix* MatrixMul(Matrix* b_matrix, Matrix* c_matrix); // 矩阵乘法
Matrix* ScalarMul(Matrix* matrix, double scalar); // 矩阵与常量相乘

// 矩阵初始化赋值
void InitMatrix(Matrix* arr, double* m){
    for (int i = 0; i < arr->rows; i++){
        for (int j = 0; j < arr->cols; j++){
            arr->mat[i*arr->cols+j] = m[i*arr->cols+j];
        }
    }
}
// 打印矩阵
void PrintMatrix(Matrix* matrix){
    for (int i = 0; i < matrix->rows; i++){
        for (int j = 0; j < matrix->cols; j++){
            printf("%.3lf ", matrix->mat[i*matrix->cols+j]);
        }
        printf("\n");
    }
}
// 复制矩阵
Matrix* copyMatrix(Matrix* matrix){
    NewMatrix(result, matrix->rows, matrix->cols, matrix->mat);
    return result;
}
// 转置矩阵
Matrix* TransMatrix(Matrix* matrix){
    MallocMatrix(result, matrix->cols, matrix->rows);
    int index0=0,index1=0;
    for (int i = 0; i < result->rows; i++){
        index0=i;
        for (int j = 0; j < result->cols; j++){
            result->mat[index1++]=matrix->mat[index0];
            index0+=matrix->cols;
        }
    }
    return result;
}

// 矩阵加减法(含广播机制)
Matrix* MatrixAdd(Matrix* b_matrix, Matrix* c_matrix){
    int b_rows = b_matrix->rows;
    int b_cols = b_matrix->cols;
    int c_rows = c_matrix->rows;
    int c_cols = c_matrix->cols;

    // 判断是否满足广播机制的条件
    if (!((b_rows == c_rows || b_rows == 1 || c_rows == 1) &&
          (b_cols == c_cols || b_cols == 1 || c_cols == 1))) {
        printf("Error: Cannot perform matrix addition/subtraction with the given dimensions.\n");
        return NULL;
    }

    // 计算广播后的结果矩阵的行数和列数
    int result_rows = (b_rows > c_rows) ? b_rows : c_rows;
    int result_cols = (b_cols > c_cols) ? b_cols : c_cols;
    // 执行广播机制，将维度不同的矩阵扩展到相同的维度
    MallocMatrix(result, result_rows, result_cols); // 网络运算同时输入3组及以上的数据进行矩阵运算时该行会出错，原因暂时未知
    for (int i = 0; i < result_rows; i++){
        for (int j = 0; j < result_cols; j++){
            // 计算广播后的索引
            int b_row_index = (b_rows == 1) ? 0 : i % b_rows;
            int b_col_index = (b_cols == 1) ? 0 : j % b_cols;
            int c_row_index = (c_rows == 1) ? 0 : i % c_rows;
            int c_col_index = (c_cols == 1) ? 0 : j % c_cols;
            // 执行加减法运算
            result->mat[i*result_cols+j] = b_matrix->mat[b_row_index*b_cols+b_col_index] + c_matrix->mat[c_row_index*c_cols+c_col_index];
        }
    }
    return result;
}
// 矩阵与常量相加
Matrix* ScalarAdd(Matrix* matrix, double scalar){
    MallocMatrix(result, matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++){
        for (int j = 0; j < matrix->cols; j++){
            result->mat[i * result->cols + j] = matrix->mat[i*matrix->cols+j] + scalar;
        }
    }
    return result;
}
// 矩阵乘法
Matrix* MatrixMul(Matrix* b_matrix, Matrix* c_matrix){
    int b_rows = b_matrix->rows;
    int b_cols = b_matrix->cols;
    int c_rows = c_matrix->rows;
    int c_cols = c_matrix->cols;

    if (b_cols != c_rows){
        printf("Error: Cannot perform matrix multiplay with the given dimensions.\n");
        return NULL;
    }
    int result_rows = b_rows;
    int result_cols = c_cols;
    int kmiddle = b_cols;
    MallocMatrix(result, result_rows, result_cols);
    for (int i = 0; i < result_rows; i++){
        for (int j = 0; j < result_cols; j++){
            for (int k = 0; k < kmiddle; k++){
                result->mat[i*result_rows+j]+=b_matrix->mat[i*kmiddle+k]*c_matrix->mat[j+k*result_cols];
            }
        }
    }
    return result;
}
// 矩阵与常量相乘
Matrix* ScalarMul(Matrix* matrix, double scalar){
    MallocMatrix(result, matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++){
        for (int j = 0; j < matrix->cols; j++){
            result->mat[i*result->cols+j] = matrix->mat[i*matrix->cols+j] * scalar;
        }
    }
    return result;
}
#endif
