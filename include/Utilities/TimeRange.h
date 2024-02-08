// TimeRange.h

#ifndef TIME_RANGE_H
#define TIME_RANGE_H

#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

class TimeMeasurements {
public:
    // Constructor
    TimeMeasurements();

    // Method to add a measurement
    void addMeasurement(std::chrono::steady_clock::duration measurement);

    // Method to display the contents of the vector
    void displayMeasurements() const;

    // Method to calculate the mean of measurements
    double calculateMean() const;

    // Method to calculate the standard deviation of measurements
    double calculateStandardDeviation() const;

private:
    std::vector<double> measurements;
};

#endif // TIME_RANGE_H
