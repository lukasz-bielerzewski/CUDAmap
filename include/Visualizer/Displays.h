/** @file Displays.h
 *
 * Displays
 *
 */

#ifndef DISPLAYS_H_INCLUDED
#define DISPLAYS_H_INCLUDED

#include "Defs/defs.h"
#include <QGLViewer/qglviewer.h>
#include <thread>

template <class T>
class Displays{
public:
    /// container
    std::vector<T> displays;
    /// add element
    template <class T2>
    void add(T2 const& _element){
        mutex.lock();
        for (auto& disp : displays)
            disp.mutex.lock();
        displays.resize(displays.size()+1);
//        displays.back().element = _element;
        displays.back().createDisplayList(_element);
        for (auto& disp : displays)
            disp.mutex.unlock();
        mutex.unlock();
    }

    /// update element
    template <class T2>
    void update(T2 const& _element, size_t elementNo){
        mutex.lock();
        if (displays.size()<=elementNo){
            for (auto& disp : displays)
                disp.mutex.lock();
            displays.resize(elementNo+1);
            for (auto& disp : displays)
                disp.mutex.unlock();
        }
        displays[elementNo].mutex.lock();
        displays[elementNo].createDisplayList(_element);
        displays[elementNo].mutex.unlock();
        mutex.unlock();
    }

    /// clear elements
    void clear(void){
        mutex.lock();
        displays.clear();
        mutex.unlock();
    }

    template <class T2>
    void createDisplayList(T2 const& _element, size_t elementNo){
        mutex.lock();
        displays[elementNo].mutex.lock();
        displays[elementNo].createDisplayList(_element);
        displays[elementNo].mutex.unlock();
        mutex.unlock();
    }

    /// draw elements
    void draw(void);
    /// mutex
    std::mutex mutex;// be carefull, the object is non copyable

    inline Displays(){}
};

template <class T>
void Displays<T>::draw() {
    mutex.lock();
    glPushMatrix();
    int objectNo = 0;
    for (auto& display : displays){
        display.draw();
        objectNo++;
    }
    glPopMatrix();
    mutex.unlock();
}

#endif // DISPLAYS_H_INCLUDED
