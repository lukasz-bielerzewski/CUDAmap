/** @file regression.h
 * Regression module interface
 * Dominik Belter
 */

#ifndef _REGRESSION_H_
#define _REGRESSION_H_

#include "Defs/defs.h"

namespace regression {

    /// Regression interface
    class Regression {
        public:

            /// Robot type
            enum Type {
                /// Gaussian mixture
                TYPE_GAUSSIAN_MIXTURE,
                /// Polynomial fitting
                TYPE_POLYNOMIAL
            };

            /// overloaded constructor
            Regression(const std::string _name, Type _type) : name(_name), type(_type) {}

            /// Name of the regression
            virtual const std::string& getName() const { return name; }

            /// get type
            virtual const Type& getType() const { return type; }

            /// Initialize training
            virtual void initializeTraining(void) = 0;

            /// train the model
            virtual void train() = 0;

            /// search for the best Approximation function
            virtual void train(const Eigen::MatrixXd& inputTrain, const Eigen::MatrixXd& outputTrain) = 0;

            /// compute output for trained function
            virtual double computeOutput(const Eigen::MatrixXd& input, int outNo) const = 0;

            /// compute gradient of trained function
            virtual void computeGradient(const Eigen::MatrixXd& input, Eigen::MatrixXd& grad) const = 0;

            /// store results
            virtual void storeResult(std::string filename) = 0;

            /// store results
            virtual void load(std::string filename) = 0;

            /// write summary
            virtual void writeSummary(void) = 0;

            /// write summary
            virtual void writeSummary(const Eigen::MatrixXd& inputTest, const Eigen::MatrixXd& outputTest) = 0;

            /// Virtual descrutor
            virtual ~Regression() {}

        protected:
            /// Regreession name
            const std::string name;
            /// Regreession type
            Type type;
    };
}

#endif // _REGRESSION_H_
