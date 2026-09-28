#pragma once
#include <vector>
#include "gyield.h"


template<class T>
class gyVarray{
private:
    T* buffer = nullptr;
    size_t size_buf = 0;
public:
    gyVarray() = default;

    ~gyVarray() {
        //clear();
    }

    void clear() {
        if (buffer) {
            gyCheckError(gyFree(buffer), "gpu free failed");
            buffer = nullptr;
            size_buf = 0;
        }
    }

    void resize_if(size_t new_size) {
        if (size_buf < new_size) {
            size_buf = new_size + new_size / 2;
            if (buffer) {
                gyCheckError(gyFree(buffer), "gpu free failed");
            }
            gyCheckError(gyMalloc((void**)&buffer, sizeof(T) * size_buf), "gpu malloc failed.");
        }
    }

    const T* data() const {
        return buffer;
    }
    T* data() {
        return buffer;
    }

    size_t size() const {
        return size_buf;
    }
    
};
