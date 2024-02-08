// TimeMeasurements.cpp

#include "Utilities/TimeRange.h"
#include <iostream>
#include <iomanip>

TimeMeasurements::TimeMeasurements() {
    // Constructor
}

void TimeMeasurements::addMeasurement(std::chrono::steady_clock::duration measurement) {
    // Convert duration to seconds (or any other desired unit)
    double measurementInSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(measurement).count();
    measurementInSeconds *= 1000000;
    measurements.push_back(measurementInSeconds);
}

void TimeMeasurements::displayMeasurements() const {
    std::cout << "Measurements: ";
    for (const auto& measurement : measurements) {
        std::cout << measurement << "  [µs], ";
    }
    std::cout << std::endl;
}

double TimeMeasurements::calculateMean() const {
    if (measurements.empty()) {
        return 0.0; // Handle the case where no measurements are present
    }

    double sum = 0.0;
    for (const auto& measurement : measurements) {
        sum += measurement;
    }

    return sum / measurements.size();
}

double TimeMeasurements::calculateStandardDeviation() const {
    if (measurements.size() < 2) {
        return 0.0; // Handle the case where there are not enough measurements for standard deviation
    }

    double mean = calculateMean();
    double sumSquaredDifferences = 0.0;

    for (const auto& measurement : measurements) {
        sumSquaredDifferences += std::pow(measurement - mean, 2);
    }

    return std::sqrt(sumSquaredDifferences / (measurements.size() - 1));
}
