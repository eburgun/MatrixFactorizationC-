/*
 * MFRecommender.h
 *
 * Evan Burgun
 * Programming Assignment 2
 *
 */

#ifndef MFRecommender_h
#define MFRecommender_h

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <numeric>
#include <time.h>
#include <iostream>
#include <fstream>
#include "CSR.h"

class MFRecommender
{
    public:
        MFRecommender(std::string trainFile, std::string testFile, int kValue, double lambda, double epsilon, int maxIter);
        ~MFRecommender(void);
        void changeKValue(int newK);
        void changeLambda(double newLambda);
        void trainSystem(void);
        double testMSE(void);
        double testSet(double mse);
        void testingMethod(void);

    private:
        int kVal;
        double lambdaVal;
        double epsVal;
        int iterations;
        CSR * trainingData;
        CSR * trainingTranspose;
        CSR * testingData;
        double ** pMatrix;
        double ** qMatrix;
        void createPandQWithRandom(void);
        void cleanUpPandQ(void);
        double fFunction(void);
        double funcDotProduct(double * a, double * b);
        double fNorm(double ** matrix,int dimension);

        void LS_GD(CSR * dataSet, double ** fixedMatrix, double ** solvingMatrix,double learningRate, std::string matrixId);
};


#include "MFRecommender.cpp"
#endif
