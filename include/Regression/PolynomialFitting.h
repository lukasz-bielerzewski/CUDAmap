//  @ Project : PolynomialFitting library
//  @ File Name : PolynomialFitting.h
//  @ Date : 2019-05-21
//  @ Author : Dominik Belter

#ifndef _PolynomialFitting_H
#define _PolynomialFitting_H

#include "Regression/regression.h"

class PolynomialFitting : public regression::Regression {
public:
    /// Pointer
    typedef std::unique_ptr<PolynomialFitting> Ptr;

    class PolyElement{
    public:
        PolyElement() {}
        /// input no
        std::vector<size_t> inputsNo;
        /// powers
        std::vector<int> powers;
        /// coefficient
        double coeff;
    };

    class Config{
      public:
        Config() {}
        Config(std::string configFilename);
        public:
            // input - number of features
            int inputsNo;

            // output - number of outputs
            int outputsNo;

            // length of the training vector
            int trainSize;

            // length of the testing vector
            int testSize;

            // maxCoeffs
            std::vector<int> maxCoeffs;

            // output format
            int outputType;

            // train dataset filename
            std::string trainFilename;

            // test dataset filename
            std::string testFilename;

            // verification dataset filename
            std::string verifFilename;
    };

	///constructor
    PolynomialFitting();
    ///constructor
    PolynomialFitting(PolynomialFitting::Config _config);
	///destructor
    ~PolynomialFitting();
    /// Initialize training
    void initializeTraining(void);
    /// search for the best PolynomialFitting function
    void train();
    /// search for the best Approximation function
    void train(const Eigen::MatrixXd& _inputTrain, const Eigen::MatrixXd& _outputTrain);
    /// compute output for the best polynomial
    double computeOutput(const Eigen::MatrixXd& input, int outNo) const;
    /// compute gradient of trained function
    void computeGradient(const Eigen::MatrixXd& input, Eigen::MatrixXd& grad) const;
    /// store results
    void storeResult(std::string filename);
    /// store results
    void load(std::string filename);
    /// write summary
    void writeSummary(void);
    /// write summary
    void writeSummary(const Eigen::MatrixXd& _inputTest, const Eigen::MatrixXd& _outputTest);

    /// change max coefs
    void setMaxCoefs(const std::vector<int>& _maxCoeffs);
    /// get polynomial element
    double getPolyCoefValue(size_t index);
    /// get polynomial element
    double getPolyCoefValue(const std::string& element);
    /// set polynomial element
    void setPolyCoefValue(size_t index, double value);
    /// get polynomial element
    void setPolyCoefValue(const std::string& element, double value);
    /// configure plane fitting
    void configurePlaneFitting(void);
    /// configure 2nd order fitting
    void configure2ndOrderFitting(void);
    /// create polynomial from string
    void createFromString(std::string polynomial);


private:
    /// config
    Config config;

    /// polynomial coefficients
    Eigen::MatrixXd polyCoef;

    /// train_data
    Eigen::MatrixXd inputTrain;
    /// output - train_data
    Eigen::MatrixXd outputTrain;

    /// verification data - input
    Eigen::MatrixXd inputVerif;
    /// verification data - output
    Eigen::MatrixXd outputVerif;

    /// test data - input
    Eigen::MatrixXd inputTest;
    /// test data - output
    Eigen::MatrixXd outputTest;

    /// is created from string?
    bool createdFromString;
    /// polynomial
    std::vector<std::pair<std::string,PolyElement>> polyString;

    /// compute value for the individual component of the polynomial
    double computeComponentValue(size_t elementNo, const Eigen::MatrixXd& sample) const;
    /// test GaussianMixture function
    void testResults(const std::string& filename, const Eigen::MatrixXd& _input, const Eigen::MatrixXd& _output);
    /// extract input no and power from string
    void extractInputNoAndPower(const std::string& atom, size_t& inputNo, int& power) const;
    /// print polynomial
    void printPolynomial(void);
};

namespace regression {
    /// create a regression objects
    std::unique_ptr<Regression> createPolynomialFitting(void);
    /// create a regression objects
    std::unique_ptr<Regression> createPolynomialFitting(PolynomialFitting::Config config);
}

#endif  //_PolynomialFitting_H
