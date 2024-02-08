#ifndef OGLWIDGET_H
#define OGLWIDGET_H

#include <QOpenGLWidget>
#include <QWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QImage>
#include <QVector3D>
#include <QColor>
#include <QMouseEvent>
#include <QPoint>
#include <QMatrix4x4>
#include <QMatrix3x3>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>



class OGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT
public:
    OGLWidget(QWidget* parent = nullptr);
    virtual ~OGLWidget();


protected:


    void loadImage(int imageIndex);


private:
    QImage* image = nullptr;
    QImage* depthImage = nullptr;

    int originalWidth = 1;
    int originalHeight = 1;
};

#endif // OGLWIDGET_H
