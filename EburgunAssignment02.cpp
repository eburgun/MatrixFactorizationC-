//
//  EburgunAssignment02.cpp
//
//
//  Created by Evan Burgun on 10/3/16.
//
//

#include <stdio.h>
#include "MFRecommender.h"

int main(){

    bool running = true;
    std::string outPut = "Output.txt";
    std::string trainingFile = "train1.txt";
    std::string testFile = "test1.txt";
    int kVal = 10;
    float lambdaVal = 0.01;
    float epsilon = 0.0001;
    int maxTries = 200;

    MFRecommender * rec = new MFRecommender(trainingFile,testFile,kVal,lambdaVal,epsilon,maxTries);



    std::cout << "Hello, welcome to my Recommender System!" << std::endl;
    std::cout << "Please wait while we load your data." << std::endl;




    std::cout << "Data Loaded" << std::endl;
    std::cout << "Please choose from the options below:" << std::endl;
    std::cout << "1. Define K Value. (Default == 10)" << std::endl;
    std::cout << "2. Define Lambda Value. (Default == 1.0)" << std::endl;
    std::cout << "3. Train System" << std::endl;
    std::cout << "4. Test trained System" << std::endl;
    std::cout << "5. Create Test Report" << std::endl;
    std::cout << "Q. Exit" << std::endl;

    bool recHasRun = false;
    while(running){
        std::cout << "Which option would you like to choose? " << std::endl;
        std::string userInput;
        std::cin >> userInput;
        if(userInput == "1"){
            std::string newKVal;
            std::cin >> newKVal;
            try
            {
                int kValue = atoi(newKVal.c_str());
                rec->changeKValue(kValue);
            }
            catch(int e)
            {
                std::cout << "An error has occured" <<  std::endl;
            }
            userInput = "0";
            newKVal = "0";
        }else if(userInput == "2"){
            std::string newLambda;
            std::cin >> newLambda;
            try
            {
                float lambda = strtof(newLambda.c_str(),NULL);
                rec->changeLambda(lambda);
            }
            catch(int e)
            {
                std::cout << "An error has occured" << std::endl;
            }
            userInput = "0";
        }else if(userInput == "3"){
            rec->trainSystem();
            userInput = "0";
        }else if(userInput == "4"){
            float mse = rec->testMSE();
            rec->testSet(mse);
            userInput = "0";
        }else if(userInput == "5"){
            rec->testingMethod();
            userInput = "0";
        }else if(userInput == "Q" || userInput == "q"){
            std::cout << "Have a nice day!" << std::endl;
            running = false;
        }else {
            std::cout << "I'm sorry, I didn't quite catch that." << std::endl;
        }


    }

    delete rec;

    return 0;

}
