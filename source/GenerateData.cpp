#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <string>

void writeDataTarget(std::string name, std::vector<std::vector<double>> input, std::vector<int> target, int N){
    std::ofstream outputFile(name);
    if (outputFile.is_open()) {
        for (int i = 0; i < N; ++i) {
            outputFile << 1.0 << "\t" << input[i][0] << "\t" << input[i][1] << "\t" << target[i] << "\n";
        }
        outputFile.close();
        std::cout << "Data written to: " << name << std::endl;
    }
    else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

void writeData(std::string name, std::vector<std::vector<double>> input, int N){
    std::ofstream outputFile(name);
    if (outputFile.is_open()) {
        for (int i = 0; i < N; ++i) {
            outputFile << 1.0 << "\t" << input[i][0] << "\t" << input[i][1] << "\n";
        }
        outputFile.close();
        std::cout << "Data written to: " << name << std::endl;
    }
    else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

int main(){

    // Generate Training data of inputs and targets like in book
    std::vector<std::vector<double>> inputs;
    std::vector<int> targets;

    int N = 300; // Number of training points
    std::random_device rnd1;
    std::mt19937 grnd1(rnd1());
    //std::uniform_real_distribution<double> dis(-2.0, 2.0);
    std::normal_distribution<double> disGauss(0, 0.3); // (Mean, Standard deviation)
    std::normal_distribution<double> dis(0, 1); // (Mean, Standard deviation)
    for (int i = 0; i < N; i++){
        // Generate true value of x1, x2
        double x1 = dis(grnd1);
        double x2 = dis(grnd1);
        // Generate Normal blurring
        double rndx1 = disGauss(grnd1);
        double rndx2 = disGauss(grnd1);

        int t;

        if ((x1 > -2*fabs(x2) + 0.5) && (x1 < -x2 + 1.5)){t = 1;}
        //if ((x1 > -2*sin(6*x2) + 0.5)){t = 1;}
        else{t = 0;}

        //std::vector<double> pair = {x1 + rndx1, x2 + rndx2}; // Add Normal blurring
        std::vector<double> pair = {x1, x2};

        inputs.push_back(pair);
        targets.push_back(t);
    }

    // Write training data to a text file
    writeDataTarget("training_data.txt", inputs, targets, N);

    // Make predictions for new data 
    std::random_device rnd2;
    std::mt19937 grnd2(rnd2());
    std::vector<std::vector<double>> inputs_new;
    std::vector<int> targets_new;
    for (int j = 0; j < N; ++j){
        double x1_new = dis(grnd2);
        double x2_new = dis(grnd2);
        std::vector<double> input_example = {x1_new, x2_new};

        inputs_new.push_back(input_example);
    }

    // Write new data to a text file
    writeData("new_data.txt", inputs_new, N);

return 0;
}