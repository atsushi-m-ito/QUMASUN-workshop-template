#pragma once

/**************************
* 一時メモリを毎回allocateしないようにして高速化するためのクラス
*
* 何度も呼ばれる下位の関数で一時メモリの確保と開放をしないようにする。
* そのために、上位関数で大きめのworkバッファを確保し、
* 下位の関数では一時メモリをworkの中の一部の利用権を確保してポインタだけ受け取る。
* 利用券の解法は、スコープの脱出時や下位の関数の脱出時などに纏めて行う
* バッファの確保と開放はこのクラスではやらない。
* メモリの利用権の管理はFILOを原則として、stackで管理
* ただし、checkpointを設けておいて、
* 関数の入り口でチェックポイントのセット、出口でチェックポイント以降に借りたものを纏めて開放という形式にすることで
* FILOを満たしやすくする。
*****************************/

#include <vector>


template<size_t ALIGNMENT>
class WorkArena {
    std::byte* m_buffer = nullptr;
    size_t m_buffer_size;
    
    std::vector<size_t> m_checkpoint;
    size_t m_current_used_size = 0;

public:
    WorkArena(void* buffer_, size_t buffer_size_ = INT64_MAX) :
        m_buffer((std::byte*)((((size_t)buffer_ + ALIGNMENT - 1) / ALIGNMENT)* ALIGNMENT)),
        m_buffer_size(buffer_size_),
        m_checkpoint(100)
    {
        m_checkpoint.push_back(0);
    };

    size_t AlignedSize(size_t size) {
        return ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }
    /*
    template<class T>
    T* AlignedPtrHead(void* ptr) {
        size_t BT = (sizeof(T) - 1);
        return ((std::byte*)ptr + BT) & (~BT);
    }
    */

    //T型の配列を、ALIGNでアライメントされた領域として貸し出す//
    template<class T>
    T* Suballoc(size_t count) {
        size_t size = AlignedSize(sizeof(T) * count);
        if (m_current_used_size + size > m_buffer_size) {
            printf("ERROR: WorkArena, memory range is over\n");
            return nullptr;
        }
        T* p = (T*)(m_buffer + m_current_used_size);
        m_current_used_size += size;
        return p;
    }

    //チェックポイントを設定。次ぐにReleaseをcallした時にはここまで一気に開放する//
    void CheckPoint() {
        m_checkpoint.push_back(m_current_used_size);
    }

    //前回のチェックポイント以降のメモリ貸し出しをクリアする//
    void Release() {
        if (m_checkpoint.empty())return;
        m_current_used_size = m_checkpoint.back();
        m_checkpoint.pop_back();        
    }
};



//必要なメモリサイズを計算する//
template<size_t ALIGNMENT>
class AlingedMemSizeCounter {
    size_t m_total_size = 0;
    //size_t m_alignment;
public:
    
    size_t AlignedSize(size_t size) {
        return ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }

    //型Tをcount個確保するために必要なサイズを計算して、totalに足しこむ//
    //
    template<class T>
    size_t Count(size_t count, size_t number) {
        size_t size = AlignedSize(sizeof(T) * count) * number;
        m_total_size += size;
        return size;
    }

    size_t Total() {
        return m_total_size;
    }
};

