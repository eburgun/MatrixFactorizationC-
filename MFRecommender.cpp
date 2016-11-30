/*
 * MFRecommender.cpp
 *
 * Evan Burgun
 * Programming Assignment 2
 *
 */

#include "MFRecommender.h"

MFRecommender::MFRecommender(std::string trainFile, std::string testFile, int kValue, double lambda, double epsilon, int maxIter)
{
    kVal = kValue;
    lambdaVal = lambda;
    epsVal = epsilon;
    iterations = maxIter;
    trainingData = new CSR(trainFile);
    trainingTranspose = new CSR(trainFile);
    trainingTranspose->transpose();
    testingData = new CSR(testFile);
    srand(time(NULL));
    createPandQWithRandom();

}

MFRecommender::~MFRecommender(void)
{

    cleanUpPandQ();
    delete trainingData;
    delete testingData;


}

void MFRecommender::createPandQWithRandom(void)
{
    pMatrix = new double*[trainingData->rows];
    qMatrix = new double*[trainingData->columns];
    for(int i = 0; i < trainingData->rows; i++){
        pMatrix[i] = new double[kVal];
        for(int j = 0; j < kVal; j++){
            pMatrix[i][j] = (double)rand() / (double)RAND_MAX;
        }

    }
    for(int i = 0; i < trainingData->columns; i++){
        qMatrix[i] = new double[kVal];
        for(int j = 0; j< kVal; j++){
            qMatrix[i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

void MFRecommender::cleanUpPandQ(void)
{
    for(int i = 0; i<trainingData->rows; i++){
        delete [] pMatrix[i];
    }
    for(int i = 0; i<trainingData->columns; i++){
        delete [] qMatrix[i];
    }
    delete [] pMatrix;
    delete [] qMatrix;
}

void MFRecommender::changeKValue(int newK)
{
    kVal = newK;
    cleanUpPandQ();
    createPandQWithRandom();
}

void MFRecommender::changeLambda(double newLambda)
{
    lambdaVal = newLambda;
}
//check this
double MFRecommender::fFunction(void)
{

    double pNorm = fNorm(pMatrix,trainingData->rows);
    double qNorm = fNorm(qMatrix,trainingData->columns);

    double lambdaQuantity = (pNorm + qNorm) * lambdaVal;
    double fSum = 0.0;
    for(int i = 0; i < trainingData->rows;i++)
    {
        for(int j = 0; j <  trainingData->columns;j++)
        {
            fSum += pow((trainingData->getElement(i,j) - funcDotProduct(pMatrix[i],qMatrix[j])),2);
        }
    }
    return fSum + lambdaQuantity;
}

double MFRecommender::funcDotProduct(double * a, double * b)
{
    double product = 0.0;
    for(int i = 0; i < kVal; i++)
    {
        product += a[i]*b[i];
    }
    return product;
}

double MFRecommender::fNorm(double ** matrix,int dimension)
{
    double norm = 0.0;
    for(int i = 0; i <  dimension; i++){
        for(int j = 0; j < kVal; j++){
            norm += pow(matrix[i][j],2);
        }
    }
    return norm;
}

void MFRecommender::LS_GD(CSR * dataSet, double ** fixedMatrix, double ** solvingMatrix,double learningRate,std::string matrixId)
{
    double lambdaValue = 1 - (lambdaVal * learningRate * 2);
    for(int i = 0; i < dataSet->rows;i++){
        double newItem[kVal];
        double sumMatrix[kVal];
        for(int j = dataSet->rowPtr[i]; j < dataSet->rowPtr[i+1]; j++){
            double dotProduct = 0.0;
            if(matrixId == "p"){
                dotProduct = funcDotProduct(solvingMatrix[i],fixedMatrix[dataSet->columnIndex[j]]);
            }
            else if(matrixId == "q"){
                dotProduct = funcDotProduct(fixedMatrix[dataSet->columnIndex[j]],solvingMatrix[i]);
            }
            double sumMult = (dataSet->ratingVals[j] - dotProduct);
            for(int k = 0; k < kVal; k++){
                sumMatrix[k] = fixedMatrix[dataSet->columnIndex[j]][k] * sumMult;
            }
        }
        double * newP = new double[kVal];
        for(int j = 0; j < kVal; j++){
            newItem[j] = solvingMatrix[i][j] * lambdaValue;
            sumMatrix[j] = sumMatrix[j] * (learningRate * 2);
            newP[j] = sumMatrix[j] + newItem[j];

        }
        double * temp = solvingMatrix[i];
        solvingMatrix[i] = newP;
        delete temp;
    }
}

void MFRecommender::trainSystem(void)
{
    int i = 0;
    double lastIter = 0.0;
    while(i <  iterations){
        LS_GD(trainingData, qMatrix, pMatrix, 0.025, "p");
        
        LS_GD(trainingTranspose, pMatrix, qMatrix, 0.025, "q");
        
        double curIter = fFunction();
        if(i > 0 && sqrt(pow((curIter - lastIter),2)/lastIter) < epsVal){
            break;
        } else {
          lastIter = curIter;
        }
        i++;
    }
}

double MFRecommender::testMSE(void)
{
    double mse = 0.0;
    for(int i = 0; i < testingData->rows;i++){
        for(int j = testingData->rowPtr[i]; j < testingData->rowPtr[i+1]; j++){
            double prediction = funcDotProduct(pMatrix[i],qMatrix[testingData->columnIndex[j]]);
            mse += pow(testingData->ratingVals[j] - prediction,2);
        }
    }

    mse /= testingData->nonZeroValues;
    return mse;
}

double MFRecommender::testSet(double mse)
{
    double rmse = sqrt(mse);
    std::cout << "k = ";
    std::cout << kVal;
    std::cout << " lambda = ";
    std::cout << lambdaVal;
    std::cout << " maxIters = ";
    std::cout << iterations;
    std::cout << " epsilon = ";
    std::cout << epsVal;
    std::cout << " mse = ";
    std::cout << mse;
    std::cout << " rmse = ";
    std::cout << rmse << std::endl;
    return rmse;
}

void MFRecommender::testingMethod(void)
{
    int kVals [] = {10 , 50};
    double lambVals [] = {0.01,0.1,1,10};
    int iters [] = {50,100,200};
    double epsilonVals [] = {0.0001,0.001,0.01};
    std::ofstream outfile ("results.txt");
    for(int i = 0; i < 2; i++){
        kVal = kVals[i];
        for(int j = 0; j < 4; j++){
            lambdaVal = lambVals[j];
            for (int k = 0; k < 3; k++){
                iterations = iters[k];
                for (int l = 0; l < 3; l++){
                    epsVal = epsilonVals[l];
                    cleanUpPandQ();
                    createPandQWithRandom();
                    clock_t trainStart = clock();
                    trainSystem();
                    clock_t trainFinish = clock();
                    clock_t testStart = clock();
                    double mse = testMSE();
                    double rmse = testSet(mse);
                    clock_t testFinish = clock();
                    outfile << kVal;
                    outfile << " ";
                    outfile << lambdaVal;
                    outfile << " ";
                    outfile << iterations;
                    outfile << " ";
                    outfile << epsVal;
                    outfile << " ";
                    outfile << mse;
                    outfile << " ";
                    outfile << rmse;
                    outfile << " ";
                    outfile << (double)(trainFinish - trainStart) * 1000.0/CLOCKS_PER_SEC;
                    outfile << " ";
                    outfile << (double)(testFinish - testStart) * 1000.0/CLOCKS_PER_SEC;
                    outfile << "\n";

                }
            }
        }
    }

    outfile.close();
}
