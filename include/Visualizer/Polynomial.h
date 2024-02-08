/** @file Polynomial.h
 *
 * 3D polynomial
 *
 */

#ifndef POLYNOMIAL_H_INCLUDED
#define POLYNOMIAL_H_INCLUDED

#include "Defs/defs.h"
#include "Visualizer/DisplayObject.h"
#include "Regression/regression.h"

class DisplayPolynomial : public DisplayObject{
public:
    regression::Regression* element;

    /// center
    double centerX;
    /// center
    double centerY;
    /// offset
    double offsetZ;
    /// size
    double surfWidth;
    /// size
    double surfLength;
    /// vertices no
    size_t verticesNo;

    /// construction
    DisplayPolynomial() : centerX(0.0), centerY(0.0), offsetZ(0.0), surfWidth(1.0), surfLength(1.0), verticesNo(10)
    {
        type = DisplayObjectType::TYPE_POLYNOMIAL;
    }

    /// create display list
    void createDisplayList(double _centerX, double _centerY, double _offsetZ, double _surfWidth, double _surfLength,
                           size_t _verticesNo, regression::Regression* poly);

    /// update display list
    void updateDisplayList();
};

#endif // LINE_H_POLYNOMIAL
