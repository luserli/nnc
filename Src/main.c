#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "cmatrix.h"
#include "parameter.h"

int main(void){
    Layer* model=ModelCreate(parameter_w, parameter_b);
    // ModelPrint(model);
    double x[]={0,1};
    NewMatrix(X, 1, sizeof(x)/sizeof(double), x);
    printf("Input:\n");
    PrintMatrix(X);

    Matrix* Z=predict(X,model);
    printf("Output:\n");
    PrintMatrix(Z);
    
    return 0;
}
