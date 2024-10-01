#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <Eigen/Dense> // library for algebra operations

// convert my matrix of vetors to Eigen Matrix
Eigen::MatrixXd eigenMatrix(const std::vector<std::vector<double>>& matrix);
// convert my vector into Eigen vector
Eigen::VectorXd eigenVector(std::vector<double>& vec);
// Read data from a file
std::vector<std::vector<double>> ReadData(const std::string& name, int N);
// Read training data with know classification
std::pair<std::vector<std::vector<double>>, std::vector<int>> ReadDataTraining(const std::string& name, int N);
// Write data to file for python visualization
void writeData(const std::string& name, std::vector<std::vector<double>> input, std::vector<int> target, int N);
// tmp data write
void writeDataPlot(const std::string& name, std::vector<std::vector<double>> vecData);

std::vector<std::vector<double>> readDataFromFile(const std::string& filename);

bool isFileEmpty(const std::string& filename);

bool fileExists(const std::string& filename);

std::vector<std::vector<std::vector<double>>> readWeights(const std::string& filename);

double readAlpha(const std::string& filename);

class NeuralNetwork {
private:
    // global variables in class so I can easily use them
    const int input_size;
    const int number_hidden;
    const std::vector<int> number_neurons;
    const int output_size;

    double alphaValue;

    // Weights for the neural network
    std::vector<std::vector<std::vector<double>>> weights;
    int NUM_WEIGHTS;

    // Matrices
    Eigen::MatrixXd Hessian;
    Eigen::MatrixXd InverseHessian;

    // Backpropagation + Hessian
    std::vector<std::vector<double>> hidden; // activation of hidden units z = h(a)
    std::vector<std::vector<double>> hidden_a; // input for activation a
    std::vector<double> output_a;
    std::vector<double> output_z;

    // Learning rate
    const double learning_rate;

    
    // Activation function (tanh) - I dont want because of projection (-1, 1) and its messy in ln(-1)
    //double tanh_activation(double x) {
    //    return tanh(x);
    //}

    // Derivative of tanh function
    //double tanh_derivative(double x) {
    //    return 1.0 - tanh(x) * tanh(x);
    //}

    // Activation function sigmoid!!!! - using same name so I dont have to change all in code
    double tanh_activation(double x){
        return 1/(1 + exp(-x));
    }

    //Derivative of sigmoid
    double tanh_derivative(double x) {
        return tanh_activation(x)*(1 - tanh_activation(x));
    }

    // Second derivatives of sigmoid
    double second_derivative(double x) {
        return (tanh_derivative(x) - 2*tanh_activation(x)*tanh_derivative(x));
    }

public:
    // Initialize class
    NeuralNetwork(const int input_size, const int number_hidden, const std::vector<int> number_neurons, const int output_size, const double learning_rate, double alphaValue)
        : input_size(input_size), number_hidden(number_hidden),
          number_neurons(number_neurons), output_size(output_size),
          learning_rate(learning_rate), alphaValue(alphaValue) {

        // Initialize weights and biases with random values
        initializeWeights();

        // initialize number of weights
        int count = 0;
        for (int k = weights.size() - 1; k > -1; k--){
            for (int i = 0; i < weights[k].size(); i++){
                for (int j = 0; j < weights[k][i].size(); j++){
                    count++;
                }
            }
        }
        NUM_WEIGHTS = count;
    }

    void initializeWeights(){
        weights.clear();

         // Initialize weights and biases with random new values (for new training with alpha also)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distribution(-10.0, 10.0);

        // Initialize weights
        std::vector<std::vector<double>> weights_invidual; // temporary matrix

        // Initialize weights_input_hidden
        for (int i = 0; i < number_neurons[0]; ++i) {
            std::vector<double> weights_row;
            for (int j = 0; j < input_size; ++j) {
                weights_row.push_back(distribution(gen));
            }
            weights_invidual.push_back(weights_row); // first column are biases
        }
        weights.push_back(weights_invidual);

        // Initialize weights_hidden_hidden
        if (number_neurons.size() > 1 ){
            for (int k = 0; k < number_neurons.size() - 1; k++){
                weights_invidual.clear(); // temporary matrix
                for (int i = 0; i < number_neurons[k + 1]; ++i) {
                    std::vector<double> weights_row;
                    for (int j = 0; j < number_neurons[k] + 1; ++j) { // extra bias "+1"
                        weights_row.push_back(distribution(gen));
                    }
                    weights_invidual.push_back(weights_row); // first column are biases
                }
                weights.push_back(weights_invidual);
            }
        }

        // Initialize weights_hidden_output
        weights_invidual.clear(); // temporary matrix
        for (int i = 0; i < output_size; ++i) {
            std::vector<double> weights_row;
            for (int j = 0; j < number_neurons[number_neurons.size() - 1] + 1; ++j) { // extra bias "+1"
                weights_row.push_back(distribution(gen));
            }
            weights_invidual.push_back(weights_row); // first column are biases
        }
        weights.push_back(weights_invidual);
    }

    void setMyWeights(std::vector<std::vector<std::vector<double>>> myWeightsData){
        weights = myWeightsData;
    }

    void setMyAlphaValue(double alp){
        alphaValue = alp;
    }

    int numberOfWeightsFunc(){ // return number of weights
        return NUM_WEIGHTS;
    }

    void printWeights(){ // prints weights in main()
        for (int k = 0; k < weights.size(); k++){
            std::cout << std::endl;
            for (int i = 0; i < weights[k].size(); i++){
                for (int j = 0; j < weights[k][i].size(); j++){
                    std::cout << weights[k][i][j] << "  ";
                }
                std::cout << std::endl;
            }
        }
    }

    // Forward pass thru NN
    std::vector<double> forward(const std::vector<double>& input) {

        std::vector<double> output(output_size, 0.0);

        hidden.clear(); // activation of hidden units z = h(a)
        hidden_a.clear(); // input for activation a
        output_a.clear();
        output_z.clear();
        for (int k = 0; k < weights.size(); k++){
            std::vector<double> tmp(weights[k].size(), 0.0);
            std::vector<double> tmp_a(weights[k].size(), 0.0);
            if (k == 0){
                for (int i = 0; i < number_neurons[k]; ++i) {
                    for (int j = 0; j < input_size; ++j) {
                        tmp[i] += input[j] * weights[k][i][j];
                    }
                    tmp_a[i] = tmp[i];
                    tmp[i] = tanh_activation(tmp[i]);
                }
                tmp.insert(tmp.begin(), 1.0); // add to the begining of a vector 1 for biases in  next layer
                hidden.push_back(tmp);
                hidden_a.push_back(tmp_a);
            }
            else if (k < weights.size() - 1){
                for (int i = 0; i < number_neurons[k]; ++i) { 
                    for (int j = 0; j < number_neurons[k-1] + 1; ++j) { // extra bias: "+1"
                        tmp[i] += hidden[k - 1][j] * weights[k][i][j];
                    }
                    tmp_a[i] = tmp[i];
                    tmp[i] = tanh_activation(tmp[i]);
                }
                tmp.insert(tmp.begin(), 1.0); // add to the begining of a vector 1 for biases in  next layer
                hidden.push_back(tmp);
                hidden_a.push_back(tmp_a);
            }
            else {
                for (int i = 0; i < output_size; ++i) {
                    for (int j = 0; j < number_neurons[k-1] + 1; ++j) { // extra bias: "+1"
                        tmp[i] += hidden[k - 1][j] * weights[k][i][j];
                        output[i] += hidden[k - 1][j] * weights[k][i][j];
                    }
                    tmp_a[i] = tmp[i];
                    tmp[i] = tanh_activation(tmp[i]);

                    output[i] = tanh_activation(output[i]);
                }
                output_z = tmp;
                output_a = tmp_a;
            }
        }
        return output;
    }

    // Backward pass (backpropagation)
    void backward(const std::vector<double>& input, int target) {
        // Forward pass
        std::vector<double> output(output_size, 0.0);
        output = forward(input);

        // Backward pass
        std::vector<std::vector<double>> deltas;
        // Calculate output layer error
        std::vector<double> output_errors(output_size, 0.0);
        for (int i = 0; i < output_size; ++i) {
            output_errors[i] = output[i] - target;
        }

        deltas.push_back(output_errors);
        
        // Calculate hidden units errors
        for (int k = 0; k < weights.size(); k++){

            if (k == 0){ // Error output - last hidden
                std::vector<double> tmp(number_neurons[number_neurons.size() - 1], 0.0);

                for (int i = 0; i < number_neurons[number_neurons.size() - 1]; ++i) {
                    for (int j = 0; j < output_size; ++j) {
                        tmp[i] += deltas[k][j] * weights[weights.size() - 1][j][i+1];
                    }
                    tmp[i] *= tanh_derivative(hidden_a[hidden_a.size()-1][i]);
                }
                deltas.push_back(tmp);
            }
            else if (k < weights.size() - 1) { // Error in hiddens
                std::vector<double> tmp(number_neurons[number_neurons.size() - 1 - k], 0.0);

                for (int i = 0; i < number_neurons[number_neurons.size() - 1 - k]; i++){ 
                    for (int j = 0; j < number_neurons[number_neurons.size() - k]; ++j) {
                        tmp[i] += deltas[k][j] * weights[weights.size() - 1 - k][j][i+1];
                    }
                    tmp[i] *= tanh_derivative(hidden_a[hidden_a.size() - 1 - k][i]);
                }
                deltas.push_back(tmp);
            }
        }

        // Update weights with alpha*w.T*w term
        for (int k = 0; k < weights.size(); k++){
            if (k == 0){
                for (int i = 0; i < output_size; ++i) {
                    for (int j = 0; j < number_neurons[number_neurons.size() - 1]; ++j) {
                        weights[weights.size() - 1][i][j+1] -= learning_rate * (deltas[k][i] * hidden[hidden.size()-1][j+1] + alphaValue * weights[weights.size() - 1][i][j+1]);
                    }
                    weights[weights.size() - 1][i][0] -= learning_rate * (deltas[k][i] + alphaValue * weights[weights.size() - 1][i][0]); // update bias separately
                }
            }
            else if (k < weights.size() - 1){
                for (int i = 0; i < number_neurons[number_neurons.size() - k]; ++i) {
                    for (int j = 0; j < number_neurons[number_neurons.size() - 1 - k]; ++j) {
                        weights[weights.size() - 1 - k][i][j+1] -= learning_rate * (deltas[k][i] * hidden[hidden.size() - 1 - k][j+1] + alphaValue * weights[weights.size() - 1 - k][i][j+1]);
                    }
                    weights[weights.size() - 1 - k][i][0] -= learning_rate * (deltas[k][i] + alphaValue * weights[weights.size() - 1 - k][i][0]); // update bias separately
                }
            }
            else {
                for (int i = 0; i < number_neurons[number_neurons.size() - k]; ++i) {
                    for (int j = 0; j < input_size - 1; ++j) {
                        weights[weights.size() - 1 - k][i][j+1] -= learning_rate * (deltas[k][i] * input[j+1] + alphaValue * weights[weights.size() - 1 - k][i][j+1]);
                    }
                    weights[weights.size() - 1 - k][i][0] -= learning_rate * (deltas[k][i] + alphaValue * weights[weights.size() - 1 - k][i][0]); // update bias separately
                }
            }  
        }
    }

    // Return one vector weights
    Eigen::VectorXd weightsVector(){
        std::vector<double> weightsV;
        for (int k = 0; k <  weights.size(); k++){
            for (int i = 0; i < weights[k].size(); i++){
                for (int j = 0; j < weights[k][i].size(); j++){
                    weightsV.push_back(weights[k][i][j]);
                }
            }
        }
        Eigen::VectorXd wv = eigenVector(weightsV);
        return wv;
    }

    // Calculate HESSIAN by Outer-product approximation
    std::vector<std::vector<double>> outerProduct(const std::vector<double>& a, const std::vector<double>& b, double outp) {
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(b.size(), 0.0));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                result[i][j] = outp*(1-outp) * a[i] * b[j];
            }
        }
        return result;
    }

    // Function to calculate the gradient
    std::vector<double> calculateGradient(const std::vector<double>& input_hidden_a, const std::vector<double>& input_hidden) {
        std::vector<double> gradient;
        for (int j = 0; j < input_hidden_a.size(); j++){
            for (int l = 0; l < input_hidden.size(); ++l) {
                gradient.push_back(tanh_derivative(input_hidden_a[j])*input_hidden[l]);
            }
        }
        return gradient;
    }

    // Print Hessian matrix, other matrices also
    void printHessian(std::vector<std::vector<double>> hess){
        std::cout << std::endl;
        for (int i = 0; i < hess.size(); i++){
            for (int j = 0; j < hess.size(); j++){
                std::cout << hess[i][j] << "  ";
            }
            std::cout << std::endl;
        }
    }

    // Making gradient vector
    std::vector<double> returnGradient(const std::vector<double>& input_data, double output){
        std::vector<double> gradient;

        for (int k = 0; k < weights.size(); k++){
            if (k == 0){
                std::vector<double> tmp(hidden_a[k].size()*input_data.size(), 0.0);
                tmp = calculateGradient(hidden_a[k], input_data);
                gradient.insert(gradient.end(), tmp.begin(), tmp.end());
            }
            else if (k < weights.size() - 1){
                std::vector<double> tmp(hidden_a[k].size()*hidden[k-1].size(), 0.0);
                tmp = calculateGradient(hidden_a[k], hidden[k-1]);
                gradient.insert(gradient.end(), tmp.begin(), tmp.end());
            }
            else{
                std::vector<double> tmp(output_a.size()*hidden[k-1].size(), 0.0);
                tmp = calculateGradient(output_a, hidden[k-1]);
                gradient.insert(gradient.end(), tmp.begin(), tmp.end());
            }
        }
        return gradient;
    }

    Eigen::MatrixXd calculateHessian(const std::vector<double>& input_data, double output, Eigen::MatrixXd previousHessian) {
        
        // using 5.4.2 chapter for calculating Hessian
        std::vector<double> gradient = returnGradient(input_data, output);

        std::vector<std::vector<double>> bTb = outerProduct(gradient, gradient, output);
        Eigen::MatrixXd bTbMatrix = eigenMatrix(bTb);
        Eigen::MatrixXd hessian = previousHessian + bTbMatrix;

        return hessian;
    }

    Eigen::MatrixXd calculateInverseHessian(const std::vector<double>& input_data, double output, Eigen::MatrixXd previousHessian) { // inverse of Hessian (alpha*I + H)^-1 = A
        // using 5.4.2 chapter for calculating Inverse Hessian

        Eigen::MatrixXd inverseHessian;

        std::vector<double> gradient = returnGradient(input_data, output);
        Eigen::VectorXd gradientVec = eigenVector(gradient); // row vector

        // Bishop, equation (5.89) - computation inverse of Hesian by outer product approximation
        std::vector<std::vector<double>> bTb = outerProduct(gradient, gradient, output);
        Eigen::MatrixXd bTbMatrix = eigenMatrix(bTb);

        Eigen::MatrixXd upperFraction = previousHessian * bTbMatrix * previousHessian;

        double valueLowerFraction = 1 + gradientVec.transpose() * previousHessian * gradientVec;

        Eigen::MatrixXd wholeFraction = upperFraction/valueLowerFraction;

        inverseHessian = previousHessian - wholeFraction;
        
        return inverseHessian;
    }

    // Calculate final Hessian and Inverse Hessian
    void calculateHessInvHess(std::vector<std::vector<double>> inputs){

        // Need to get ourputs for training NN to calculate Hessian
        std::vector<double> outputs(inputs.size(), 0.0);
        for (int i = 0; i < inputs.size(); i++){
            double tmpp = predictSmooth(inputs[i]);
            outputs[i] = tmpp;
        }
        double Value = alphaValue;
        Eigen::MatrixXd alphaIdentity = Value * Eigen::MatrixXd::Identity(NUM_WEIGHTS, NUM_WEIGHTS);
        Eigen::MatrixXd alphaInverseIdentity = 1/Value * Eigen::MatrixXd::Identity(NUM_WEIGHTS, NUM_WEIGHTS);
        Eigen::MatrixXd tmpInverseHessian(NUM_WEIGHTS, NUM_WEIGHTS);
        Eigen::MatrixXd tmpHessian(NUM_WEIGHTS, NUM_WEIGHTS);
        InverseHessian.resize(NUM_WEIGHTS, NUM_WEIGHTS);
        Hessian.resize(NUM_WEIGHTS, NUM_WEIGHTS);


        // Calculation data by data and updating Hessians
        for (int d = 0; d < inputs.size(); d++){ 
            // Hessian & Inverse Hessian
            if (d == 0){
                tmpHessian = calculateHessian(inputs[d], outputs[d], alphaIdentity);
                tmpInverseHessian = calculateInverseHessian(inputs[d], outputs[d], alphaInverseIdentity);
            }
            else{
                InverseHessian = calculateInverseHessian(inputs[d], outputs[d], tmpInverseHessian);
                tmpInverseHessian = InverseHessian;

                Hessian = calculateHessian(inputs[d], outputs[d], tmpHessian);
                tmpHessian = Hessian;
            }

            //if ((d+1)%100 == 0){
            //    std::cout << "x100 data consumed" << std::endl;
            //}
        }
    }

    void upgradeAlpha(const std::vector<std::vector<double>>& inputs){
        calculateHessInvHess(inputs);
        // Compute eigenvalues
        Eigen::MatrixXd MyHess2 = Hessian - alphaValue*Eigen::MatrixXd::Identity(NUM_WEIGHTS, NUM_WEIGHTS);
        Eigen::EigenSolver<Eigen::MatrixXd> solver(MyHess2);
        // Get the eigenvalues
        Eigen::VectorXd eigenvalues = solver.eigenvalues().real();

        // Update alpha
        double gamma = 0;
        for (int k = 0; k < NUM_WEIGHTS; k++){
            gamma += eigenvalues(k)/(alphaValue + eigenvalues(k));
        }

        Eigen::VectorXd weightsVec = weightsVector();
        
        alphaValue = gamma/(weightsVec.transpose() * weightsVec);
    }

    // Train the neural network
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<int> targets, double num) {
        bool check = true;
        double value;
        double lowest;
        double Error;

        // initialize random weights
        initializeWeights();
        upgradeAlpha(inputs);
        std::cout << "Alpha initial = " << alphaValue << std::endl;
        // START TO TRAIN
        do {
            for (int i = 0; i < inputs.size(); i++){
                backward(inputs[i], targets[i]); // this updates weights by training
            }

            // Compute error
            //calculateHessInvHess(inputs); // for Inverse Hessian
            Error = 0;
            for (int i = 0; i < inputs.size(); i++){
                std::vector<double> outputtt;
                //outputtt.push_back(predictBayesSmooth(inputs[i], InverseHessian));
                outputtt.push_back(predictSmooth(inputs[i]));
                // Errors
                for (int j = 0; j < output_size; j++){
                    Error += -(targets[i]*log(outputtt[j]) + (1 - targets[i])*log(1 - outputtt[j]));
                }
            }

            Eigen::VectorXd weightsVec = weightsVector();
            // Also calculate whole Error of dataset
            Error += (alphaValue/2)*weightsVec.transpose()*weightsVec;

            if (check){
                // this if is here only to set my lowest value of error as my first value of error
                value = Error;
                check = false;
                lowest = value;
                std::cout << value << std::endl;
            }
            else {
                value = Error;
                std::cout << value << std::endl;
                if (value < lowest){
                    lowest = value;
                }
                else{
                    std::cout << "Alpha value = " << alphaValue << std::endl;
                    break;
                }
            }

            // upgrade alpha for next training
            upgradeAlpha(inputs);

        } while (fabs(value) >= num); // num = value at wich this whole stops (convergence value)
    }

    Eigen::MatrixXd returnHessian(){
        return Hessian;
    }

    Eigen::MatrixXd returnInverseHessian(){
        return InverseHessian;
    }

    // Predict the class for a given input
    int predict(const std::vector<double>& input){
        std::vector<double> output = forward(input);
        return (output[0] < 0.5) ? 0 : 1; // Assuming a threshold of 0.5 for binary classification (if output > 0.5 -> 1, else -> 0)
    }
    // Return output function in raw form. Not binary classification
    double predictSmooth(const std::vector<double>& input){
        std::vector<double> output = forward(input);
        return output[0];
    }

    double returnAlpha(){
        return alphaValue;
    }

    void saveWeights(){
        std::string filename = "./inputs/weights.txt";
        std::ofstream outputFile(filename, std::ios::out); // Takes care of my file weights.txt

        if (!outputFile.is_open()) {
            std::cout << "Error: Cannot open the file: " << filename << std::endl;
            return;
        }

        // Iterate over each matrix
        for (const auto& matrix : weights) {
            // Iterate over each row in the matrix
            for (const auto& row : matrix) {
                // Iterate over each value in the row and write it to the file
                for (double value : row) {
                    outputFile << value << " ";
                }
                outputFile << std::endl; // Add newline after each row
            }
            outputFile << std::endl; // Add empty line after each matrix
        }

        outputFile.close();
        std::cout << "Weights has been written to: " << filename << std::endl;
    }

    void saveAlpha(){
        std::string filename = "./inputs/alpha.txt";
        std::ofstream outputFile(filename, std::ios::out); // Takes care of my file weights.txt

        if (!outputFile.is_open()) {
            std::cout << "Error: Cannot open the file: " << filename << std::endl;
            return;
        }

        outputFile << alphaValue;

        outputFile.close();
        std::cout << "Alpha has been written to: " << filename << std::endl;
    }

};

// Write data of boundry decition for python visualization
void write_decition_boundary(const std::string& name, NeuralNetwork& neuralNetwork, std::vector<std::vector<double>> vecDat);

int main() {
    // LOADING INPUT DATA ABOUT NN AND WRITING IT INTO CONSOLE
    //if (!fileExists("./inputs/weights.txt") || isFileEmpty("./inputs/weights.txt")){
    std::cout << std::endl;
    // Load input parameters
    std::vector<std::vector<double>> inputData = readDataFromFile("./inputs/input.txt");

    const int trainOrPredict = int(inputData[6][0]);

    std::cout << "#########################################################" << std::endl;

    const int input_size = int(inputData[0][0]); //(bias, x1, x2)
    std::cout << "Input size = " << input_size << std::endl;

    const std::vector<int> number_neurons(inputData[1].begin(), inputData[1].end());
    std::cout << "Number of neurons: ";
    for (int i = 0; i < number_neurons.size(); i++){
        std::cout << number_neurons[i] << "  ";
    }
    std::cout << std::endl;

    const int number_hidden = number_neurons.size(); // Number of hidden layers
    std::cout << "Number of hidden layers = " << number_hidden << std::endl;

    const int output_size = int(inputData[2][0]); // Binary classification (1 or 0)
    std::cout << "Output size = " << output_size << std::endl;

    const double learning_rate = inputData[3][0];
    std::cout << "Learning rate = " << learning_rate << std::endl;

    const double num = inputData[5][0]; // Precition of Error where I want to end
    std::cout << "Num = " << num << std::endl;

    const double alpha = inputData[4][0];
    std::cout << "Alpha = " << alpha << std::endl;

    // Create a neural network
    NeuralNetwork neuralNetwork(input_size, number_hidden, number_neurons, output_size, learning_rate, alpha);

    std::cout << "#########################################################" << std::endl << std::endl;

    if (trainOrPredict == 1 && fileExists("./inputs/weights.txt") && !isFileEmpty("./inputs/weights.txt")
        && fileExists("./inputs/alpha.txt") && !isFileEmpty("./inputs/alpha.txt")){
        
        std::cout << "Start of prediction NN." << std::endl;

        // Read my pre-trained weights and load them into NN
        std::vector<std::vector<std::vector<double>>> myWeights = readWeights("./inputs/weights.txt");
        double myAlpha = readAlpha("./inputs/alpha.txt");
        neuralNetwork.setMyWeights(myWeights);
        neuralNetwork.setMyAlphaValue(myAlpha);
        std::cout << "Weights and alpha are loaded." << std::endl;
        //neuralNetwork.printWeights();

        // Predict new data
        std::vector<std::vector<double>> inputs_new;
        inputs_new = ReadData("./datas/new_data.txt", input_size); // 10 data values with bias
        std::vector<int> targets_new;

        for (int i = 0; i < inputs_new.size(); i++){
            int predicted_class = neuralNetwork.predict(inputs_new[i]);
            targets_new.push_back(predicted_class);
        }
        // load whole data for boundary
        std::vector<std::vector<double>> inputs_boundary;
        inputs_boundary = ReadData("./datas/whole_data.txt", input_size);

        std::cout << "New predictions made." << std::endl << std::endl;

        // Write new data to a text file
        std::cout << "#########################################################" << std::endl;
        writeData("./datas/predicted_data.txt", inputs_new, targets_new, inputs_new.size());
        write_decition_boundary("./datas/boundery_decition.txt", neuralNetwork, inputs_new);

    }
    else{
        std::cout << "Start of training NN." << std::endl << std::endl;
        std::cout << "#########################################################" << std::endl << std::endl;

        // Load training data 
        std::vector<std::vector<double>> inputs;
        std::vector<int> targets;

        std::tie(inputs, targets) = ReadDataTraining("./datas/training_data.txt", input_size); // 10 data values with bias

        // Train the neural network
        neuralNetwork.train(inputs, targets, num);
        // Save my trained weights into file
        neuralNetwork.saveWeights();
        neuralNetwork.saveAlpha();

        // Predict on training data while my NN is already trained
        std::vector<int> targets_new;
        for (int i = 0; i < inputs.size(); i++){
            int predicted_class = neuralNetwork.predict(inputs[i]);
            targets_new.push_back(predicted_class);
        }

        // Write new data to a text file
        std::cout << "#########################################################" << std::endl;
        writeData("./datas/predicted_trained_data.txt", inputs, targets_new, inputs.size());

        // Write data for boundary decition
        write_decition_boundary("./datas/boundery_decition_trained_data.txt", neuralNetwork, inputs);
    }

    std::cout << std::endl;
    
    return 0;
}

void writeData(const std::string& name, std::vector<std::vector<double>> input, std::vector<int> target, int N){
    std::ofstream outputFile(name, std::ios::out);
    if (outputFile.is_open()) {
        for (int i = 0; i < N; ++i) {
            outputFile << input[i][1] << "\t" << input[i][2] << "\t" << target[i] << "\n";
        }
        outputFile.close();
        std::cout << "Data written to: " << name << std::endl;
    }
    else {
        std::cout << "Unable to open file for writing." << std::endl;
    }
}

void write_decition_boundary(const std::string& name, NeuralNetwork& neuralNetwork, std::vector<std::vector<double>> vecDat){
    std::ofstream decisionBoundaryFile(name, std::ios::out);
    if (decisionBoundaryFile.is_open()) {
        for (int i = 0; i < vecDat.size(); i++) { // filling grid
            std::vector<double> input = vecDat[i];
            double decision_boundary_value = neuralNetwork.predictSmooth(input);
            for (int l = 0; l < input.size(); l++){
                decisionBoundaryFile << input[l] << "\t";
            }
            decisionBoundaryFile << decision_boundary_value << "\n";
        }
    decisionBoundaryFile.close();
    std::cout << "Decision boundary data written to: " << name << std::endl;
    } 
    else {
        std::cout << "Unable to open file for writing decision boundary." << std::endl;
    }
}

std::vector<std::vector<double>> ReadData(const std::string& name, int N){
    // Open the file
    std::ifstream inputFile(name);

    // Check if the file is open
    if (!inputFile.is_open()) {
        std::cout << "Error: Cannot open the file: " << name << std::endl;
        return {};
    }

    // Vector to store data
    std::vector<std::vector<double>> doubleData;

    // Read data from the file
    double value;
    while (inputFile >> value) {
        // Create a vector to store the current line of double data
        std::vector<double> lineData;

        // add bias
        lineData.push_back(value);
        
        // Read the data except the bias! N = number of data values exept bias
        for (int i = 0; i < N; i++){
            if (i == (N - 1)){
                inputFile >> value;
            }
            else{
                inputFile >> value;
                lineData.push_back(value);
            }
        }

        // Add the data to the vectors
        doubleData.push_back(lineData);
    }

    // Close the file
    inputFile.close();

    return doubleData;
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> ReadDataTraining(const std::string& name, int N) {
    // Open the file
    std::ifstream inputFile(name);

    // Check if the file is open
    if (!inputFile.is_open()) {
        std::cout << "Error: Cannot open the file: " << name << std::endl;
        return {{}, {}};
    }

    // Vectors to store data
    std::vector<std::vector<double>> doubleData;
    std::vector<int> intData;

    // Read data from the file
    double value;
    int intValue;
    while (inputFile >> value) {
        // Create a vector to store the current line of double data
        std::vector<double> lineData;

        //read bias
        lineData.push_back(value);
        // Read the data except the bias! N = number of data values exept bias
        for (int i = 0; i < N - 1; i++){
            inputFile >> value;
            lineData.push_back(value);
        }

        // Read the integer value
        inputFile >> intValue;

        // Add the data to the vectors
        doubleData.push_back(lineData);
        intData.push_back(intValue);
    }

    // Close the file
    inputFile.close();

    return {doubleData, intData};
}

// Convert to Eigen::MatrixXd
Eigen::MatrixXd eigenMatrix(const std::vector<std::vector<double>>& matrix){
    Eigen::MatrixXd eigenMatrix(matrix.size(), matrix.size());
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix.size(); ++j) {
            eigenMatrix(i, j) = matrix[i][j];
        }
    }
return eigenMatrix;
}
// Convert to Eigen::VectorXd
Eigen::VectorXd eigenVector(std::vector<double>& vec){
    Eigen::Map<Eigen::VectorXd> eigenVectorReturn(vec.data(), vec.size());
    return eigenVectorReturn;
}

std::vector<std::vector<double>> readDataFromFile(const std::string& filename) {
    std::ifstream inputFile(filename);

    if (!inputFile.is_open()) {
        std::cout << "Error opening the file: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(inputFile, line)) {
        // Ignore everything after '#'
        size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }

        // Trim leading and trailing whitespaces
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        if (!line.empty()) {
            std::istringstream iss(line);
            double value;
            std::vector<double> numbers;

            // Read numbers from each line
            while (iss >> value) {
                numbers.push_back(value);
            }

            // Process the data as needed
            data.push_back(numbers);
        }
    }

    inputFile.close();
    std::cout << "Inputs for NN are loaded." << std::endl;
    return data;
}

bool isFileEmpty(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    return file.tellg() == 0;
}

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

std::vector<std::vector<std::vector<double>>> readWeights(const std::string& filename) {
    std::ifstream inputFile(filename);
    std::vector<std::vector<std::vector<double>>> data;

    if (!inputFile.is_open()) {
        std::cout << "Error: Cannot open the file: " << filename << std::endl;
        std::cout << "Plese check if you already trained NN and you have saved file: weights.txt" << filename << std::endl;
        return data;
    }

    std::string line;
    std::vector<std::vector<double>> currentMatrix;

    while (std::getline(inputFile, line)) {
        if (line.empty()) {
            // Empty line indicates the end of a matrix
            if (!currentMatrix.empty()) {
                data.push_back(currentMatrix);
                currentMatrix.clear();
            }
        } else {
            std::istringstream iss(line);
            std::vector<double> row;
            double value;

            while (iss >> value) {
                row.push_back(value);
            }

            if (!row.empty()) {
                currentMatrix.push_back(row);
            }
        }
    }

    // Add the last matrix if it's not empty
    if (!currentMatrix.empty()) {
        data.push_back(currentMatrix);
    }

    inputFile.close();
    return data;
}

double readAlpha(const std::string& filename) {
    std::ifstream inputFile(filename);
    double number;

    if (!inputFile.is_open()) {
        std::cout << "Error: Cannot open the file: " << filename << std::endl;
        return 0.0; // Return 0.0 as a default value
    }

    inputFile >> number; // Read the single number from the file

    inputFile.close();
    return number;
}

