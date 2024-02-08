//
//  @ Project : PolynomialFitting library
//  @ File Name : PolynomialFitting.cpp
//  @ Date : 2019-05-21
//  @ Author : Dominik Belter
//
//

#include "Regression/PolynomialFitting.h"
#include <tinyxml2.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <numeric>
#include <iostream>

/// A single instance of the polynomial fitting
//PolynomialFitting::Ptr polynomialFitting;

///constructor
PolynomialFitting::PolynomialFitting(PolynomialFitting::Config _config): Regression("Polynomial Fitting", TYPE_POLYNOMIAL), config(_config),
    createdFromString(false){

}

///constructor
PolynomialFitting::PolynomialFitting(void): Regression("Polynomial Fitting", TYPE_POLYNOMIAL), createdFromString(false){

}

///destructor
PolynomialFitting::~PolynomialFitting(){
}

///config class construction
PolynomialFitting::Config::Config(std::string configFilename){
    setlocale(LC_NUMERIC,"C");
    tinyxml2::XMLDocument config;
    std::string filename = "../../resources/" + configFilename;
    std::cout << filename << "\n";
    config.LoadFile(filename.c_str());
    if (config.ErrorID()){
        std::cout << "unable to load Polynomial Fitting config file.\n";
        std::cout << "Error id: " << config.ErrorID() << "\n";
    }

    tinyxml2::XMLElement * params = config.FirstChildElement( "PolynomialFitting" );
    params->FirstChildElement( "parameters" )->QueryIntAttribute("inputsNo", &inputsNo);
    params->FirstChildElement( "parameters" )->QueryIntAttribute("outputsNo", &outputsNo);
    params->FirstChildElement( "parameters" )->QueryIntAttribute("trainSize", &trainSize);
    params->FirstChildElement( "parameters" )->QueryIntAttribute("testSize", &testSize);
    params->FirstChildElement( "parameters" )->QueryIntAttribute("outputType", &outputType);

    maxCoeffs.resize(inputsNo);
    for (int dimNo=0;dimNo<inputsNo;dimNo++){
        std::string dimName = "dim" + std::to_string(dimNo);
        const tinyxml2::XMLElement* childNode = params->FirstChildElement(dimName.c_str());
        if (childNode != NULL){
            params->FirstChildElement( dimName.c_str() )->QueryIntAttribute("maxCoeff", &maxCoeffs[dimNo]);
        }
    }

    params = config.FirstChildElement( "trainingSet" );
    trainFilename = params->GetText();
    params = config.FirstChildElement( "testSet" );
    testFilename = params->GetText();
    params = config.FirstChildElement( "verificationSet" );
    verifFilename = params->GetText();
}

/// Initialize training
void PolynomialFitting::initializeTraining(void){
    size_t numberOfComponents = 1;
    for (const auto& maxPow : config.maxCoeffs)
        numberOfComponents *=(maxPow+1);
    polyCoef = Eigen::MatrixXd::Zero(numberOfComponents, config.outputsNo);
}

/// print polynomial
void PolynomialFitting::printPolynomial(void){
    std::cout << "From string: f(x)=";
    for (size_t elemNo=0; elemNo<(size_t)polyCoef.rows(); elemNo++){
        std::cout << polyString[elemNo].second.coeff << "*" << polyString[elemNo].first << " + ";
    }
    std::cout << "\n";

    std::cout << "From parsed values: f(x)=";
    for (size_t elemNo=0; elemNo<(size_t)polyCoef.rows(); elemNo++){
        std::cout << polyString[elemNo].second.coeff << "*";
        for (size_t inputIdx=0; inputIdx<polyString[elemNo].second.inputsNo.size(); inputIdx++){
            size_t inNo = polyString[elemNo].second.inputsNo[inputIdx];
            int power = polyString[elemNo].second.powers[inputIdx];
            std::cout << "x_" << inNo << "^" << power;
        }
        std::cout << " + ";
    }
    std::cout << "\n";
}

/// compute output for the best polynomial
double PolynomialFitting::computeOutput(const Eigen::MatrixXd& input, int outNo) const{
    double out(0.0);
    //c_1*x^0*y^0+c_2*x^1*y^0+c_3*x^2*y^0+
    //c_4*x^0*y^1+c_5*x^1*y^1+c_6*x^2*y^1+
    //c_7*x^0*y^2+c_8*x^1*y^2+c_9*x^2*y^2
    size_t numberOfComponents = polyCoef.rows();
    std::vector<double> components(numberOfComponents,1);
    size_t coeffNo=0;
    if (createdFromString){
        for (size_t elemNo=0; elemNo<(size_t)polyCoef.rows(); elemNo++){
            components[elemNo]=polyString[elemNo].second.coeff*computeComponentValue(elemNo,input);
        }
    }
    else {
        for (size_t elemNo=0; elemNo<(size_t)polyCoef.rows(); elemNo++){
            components[elemNo]=polyCoef(elemNo,outNo)*computeComponentValue(elemNo,input);
            coeffNo++;
        }
    }
    out = std::accumulate(components.begin(), components.end(),0.0);
    return out;
}

/// compute gradient of trained function
void PolynomialFitting::computeGradient(const Eigen::MatrixXd& input, Eigen::MatrixXd& grad) const{
    (void) input;
    (void) grad;
}

/// compute value for the individual component of the polynomial
double PolynomialFitting::computeComponentValue(size_t elementNo, const Eigen::MatrixXd& sample) const{
    double component = 1;

    if (!createdFromString){
        size_t elemNo=0;
        for (size_t coefNo1=0; coefNo1<(size_t)config.maxCoeffs[0]+1; coefNo1++){
            for (size_t coefNo2=0; coefNo2<(size_t)config.maxCoeffs[1]+1; coefNo2++){
                if (elemNo==elementNo){
                    component=pow(sample(0,0),(double)coefNo1)*pow(sample(0,1),(double)coefNo2);
                }
                elemNo++;
            }
        }
    }
    else{
        for (size_t inputIdx=0; inputIdx<polyString[elementNo].second.inputsNo.size(); inputIdx++){
            size_t inNo = polyString[elementNo].second.inputsNo[inputIdx];
            int power = polyString[elementNo].second.powers[inputIdx];
            component *= pow(sample(0,inNo),power);
        }
    }
    return component;
}

/// search for the best Approximation function
void PolynomialFitting::train() {
    if (!createdFromString)
        initializeTraining();
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(inputTrain.rows(), polyCoef.rows());
    Eigen::MatrixXd Vprim = Eigen::MatrixXd::Zero(polyCoef.rows(), polyCoef.rows());
    Eigen::MatrixXd p = Eigen::MatrixXd::Zero(polyCoef.rows(), config.outputsNo);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(polyCoef.rows(), polyCoef.rows());
    Eigen::MatrixXd _sample = Eigen::MatrixXd::Zero(1, config.inputsNo);
    for (int i=0;i<inputTrain.rows();i++){
        _sample = inputTrain.row(i);
        for (int j=0;j<polyCoef.rows();j++){
            V(i,j)=computeComponentValue(j,_sample);
        }
    }
    Vprim = V.transpose();
    G=Vprim*V;
    p=Vprim*outputTrain;
    G.llt().solveInPlace(p);
    polyCoef=p;
    if (createdFromString){
        for (size_t polyCoefNo=0;polyCoefNo<(size_t)polyCoef.rows();polyCoefNo++){
            polyString[polyCoefNo].second.coeff = polyCoef(polyCoefNo,0);
        }
    }
}

/// search for the best Approximation function
void PolynomialFitting::train(const Eigen::MatrixXd& _inputTrain, const Eigen::MatrixXd& _outputTrain){
    inputTrain = _inputTrain;
    outputTrain = _outputTrain;
    train();
}

/// write summary
void PolynomialFitting::writeSummary(void){
}

/// write summary
void PolynomialFitting::writeSummary(const Eigen::MatrixXd& _inputTest, const Eigen::MatrixXd& _outputTest){
    inputTest = _inputTest;
    outputTest = _outputTest;
    std::ofstream ofstr;
    ofstr.open ("polyFitt.m");
    ofstr << "close all;\n";
    for (size_t sampleNo=0;sampleNo<(size_t)inputTest.rows();sampleNo++){
        Eigen::MatrixXd input = Eigen::MatrixXd::Zero(1, 2);
        input(0,0) = inputTest(sampleNo,0); input(0,1) = inputTest(sampleNo,1);
        ofstr << "plot3(" << input(0,0) << ", " << input(0,1) << ", " << outputTest(sampleNo,0) << ",'or'); hold on\n";
        ofstr << "plot3(" << input(0,0) << ", " << input(0,1) << ", " << computeOutput(input,0) << ",'og');\n";
    }
    ofstr << "legend('reference','regression');\n";
    ofstr << "xlabel('x');\n";
    ofstr << "ylabel('y');\n";
    ofstr << "zlabel('z');\n";
    ofstr.close();
}

/// store results
void PolynomialFitting::storeResult(std::string filename){
    testResults(filename, inputTest, outputTest);
//    printPolynomial();
}

/// test GaussianMixture function
void PolynomialFitting::testResults(const std::string& filename, const Eigen::MatrixXd& _input, const Eigen::MatrixXd& _output){
    Eigen::MatrixXd _sample = Eigen::MatrixXd::Zero(1, config.inputsNo);
    int iter = 0;
    std::ofstream ofstr;
    ofstr.open (filename);
    int correct=0;
    for (int i=0;i< _input.rows();i++){
        _sample = _input.row(iter);

        ofstr << "f" << i << "(";
        for (int j=0;j<_sample.cols();j++)
            ofstr << _sample(0,j) << ",";
        ofstr << ") = (";
        for (int k=0;k<config.outputsNo;k++){
            ofstr << computeOutput(_sample,k) << ", ";
        }
        ofstr << ") should be (";
        for (int k=0;k<config.outputsNo;k++){
            ofstr << _output(iter,k) << ", ";
        }
        ofstr << ") error = (";
        for (int k=0;k<config.outputsNo;k++){
            ofstr << computeOutput(_sample,k)-(_output)(iter,k) << ", ";
        }
        ofstr << ")\n";
        double err=0;
        err = fabs(computeOutput(_sample,0)-(_output)(iter,0));
        if (err<0.5)
          correct++;
        iter++;
    }
    ofstr << "correct = " << (double(correct)/double(iter))*100 << "%\n";
    ofstr << "f(...)=";
    size_t coeffNo=0;
    for (size_t coefNo1=0; coefNo1<(size_t)config.maxCoeffs[0]+1; coefNo1++){
        for (size_t coefNo2=0; coefNo2<(size_t)config.maxCoeffs[1]+1; coefNo2++){
            ofstr << polyCoef(coeffNo,0);
            ofstr << "*x^" << coefNo1 << "*y^" << coefNo2 << "+";
            coeffNo++;
        }
    }
    ofstr << "\n";
    ofstr.close();
}

/// get polynomial element
double PolynomialFitting::getPolyCoefValue(size_t index){
    if (index<(size_t)polyCoef.rows()){
        if (createdFromString)
            return polyString[index].second.coeff;
        else
            return polyCoef(index,0);
    }
    else
        throw std::runtime_error("Polynomial fiting (getter): index out of bounds.\n");
}

/// get polynomial element
double PolynomialFitting::getPolyCoefValue(const std::string& element){
    for (const auto& elem : polyString){
        if (elem.first == element){
            return elem.second.coeff;
        }
    }
    throw std::runtime_error("No " + element + "in the polynomial\n" );
}

/// set polynomial element
void PolynomialFitting::setPolyCoefValue(size_t index, double value){
    if (index<(size_t)polyCoef.rows()){
        polyCoef(index,0) = value;
        polyString[index].second.coeff = value;
    }
    else
        throw std::runtime_error("Polynomial fiting (setter): index out of bounds.\n");
}

/// get polynomial element
void PolynomialFitting::setPolyCoefValue(const std::string& element, double value){
    for (auto& elem : polyString){
        if (elem.first == element){
            elem.second.coeff = value;
            return;
        }
    }
    throw std::runtime_error("No " + element + "in the polynomial\n" );
}

/// configure plane fitting
void PolynomialFitting::configurePlaneFitting(void){
    config.inputsNo = 2;
    config.outputsNo = 1;
    config.maxCoeffs = std::vector<int>{2,2};
    initializeTraining();
}

/// configure 2nd order fitting
void PolynomialFitting::configure2ndOrderFitting(void){
    config.inputsNo = 2;
    config.outputsNo = 1;
    config.maxCoeffs = std::vector<int>{3,3};
    initializeTraining();
}

/// change max coefs
void PolynomialFitting::setMaxCoefs(const std::vector<int>& _maxCoeffs){
    config.maxCoeffs = _maxCoeffs;
}

/// store results
void PolynomialFitting::load(std::string filename){
    (void) filename;
}

/// create polynomial from string
void PolynomialFitting::createFromString(std::string polynomial){
    std::replace(polynomial.begin(), polynomial.end(), '+', ' ');  // replace '+' by ' '

    std::vector<std::string> elements;
    std::stringstream ss(polynomial);
    std::string temp;
    while (ss >> temp)
        elements.push_back(temp);
    polyString.resize(elements.size());
    size_t elemNo=0;
    size_t maxInputsNo=0;
    for (auto& elem : elements){
        std::string elemTmp(elem);
        std::replace(elemTmp.begin(), elemTmp.end(), '*', ' ');  // replace '*' by ' '
        std::vector<std::string> atoms;
        std::stringstream ss1(elemTmp);
        std::string tempAtom;
        while (ss1 >> tempAtom){
            atoms.push_back(tempAtom);
        }

        PolyElement polyElem;
        polyElem.inputsNo.resize(atoms.size());
        polyElem.powers.resize(atoms.size());
        if (atoms.size()>maxInputsNo)
            maxInputsNo = atoms.size();
        size_t idx=0;
        for (const auto& atom : atoms){
            size_t inputNo;
            int power;
            extractInputNoAndPower(atom, inputNo, power);
            if (inputNo+1>maxInputsNo)
                maxInputsNo = inputNo+1;
            polyElem.inputsNo[idx] = inputNo;
            polyElem.powers[idx] = power;
            idx++;
        }

        polyString[elemNo] = std::make_pair(elem, polyElem);
        elemNo++;
    }


    config.inputsNo = (int)maxInputsNo;
    config.outputsNo = 1;
    polyCoef = Eigen::MatrixXd::Zero(polyString.size(), config.outputsNo);
    for (size_t polyCoefNo=0;polyCoefNo<(size_t)polyCoef.rows();polyCoefNo++){
        polyString[polyCoefNo].second.coeff = 0;
    }
    createdFromString = true;
//    printPolynomial();
}

/// extract input no and power from string
void PolynomialFitting::extractInputNoAndPower(const std::string& atom, size_t& inputNo, int& power) const{
    std::string atomTmp = atom;

    atomTmp.erase(std::remove(atomTmp.begin(), atomTmp.end(), 'x'), atomTmp.end());  // replace 'x_' by ''
    atomTmp.erase(std::remove(atomTmp.begin(), atomTmp.end(), '_'), atomTmp.end());  // replace 'x_' by ''

    std::replace(atomTmp.begin(), atomTmp.end(), '^', ' ');  // replace '^' by ' '

    std::vector<std::string> elements;
    std::stringstream ss(atomTmp);
    std::string temp;
    while (ss >> temp)
        elements.push_back(temp);
    if (elements.size()==2){
        std::string::size_type sz;
        inputNo = std::stoi(elements[0],&sz);
        power = std::stoi(elements[1],&sz);
    }
    else{
        std::cout << "something is wrong with the polynomial\n";
    }
}

std::unique_ptr<regression::Regression> regression::createPolynomialFitting(PolynomialFitting::Config config) {
    return walkers::make_unique<PolynomialFitting>(config);
}

std::unique_ptr<regression::Regression> regression::createPolynomialFitting(void) {
    return walkers::make_unique<PolynomialFitting>();
}
