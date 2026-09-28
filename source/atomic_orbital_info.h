#pragma once

/*
* 単原子の解を求めるのに必要な(ちょうどよい)軌道数を返す
* valence electronに依存する。
* 例えばSiの場合には、
* ve=4なら、3s, 3px, 3py, 3pzの4が返るが
* ve=12なら先に加えて2s, 2px, 2py, 2pの8が返る
* 
* ただし、4f軌道はスキップする仕様(一般的なタングステンなどの擬ポテンシャルでは4fはFrozenコア扱い)//
*/
std::vector<int> GetNumValenceAnglar(int Z, int ve) {
    

    if (Z < 72) {
        const std::vector<int> anglar_list{/*1s*/0, /*2s*/0, /*2p*/1, /*3s*/0, /*3p*/1, /*4s*/0, /*3d*/2, /*4p*/1,
            /*5s*/0, /*4d*/2, /*5p*/1, /*6s*/0, /*5d*/2, /*6p*/1, /*7s*/0, /*6d*/2 };

        std::vector<int> ve_anglar_list;

        const int num_occupied = Z - ve;
        int num = 0;
        const int64_t count = anglar_list.size();

        int64_t i = 0;
        for (; i < count; ++i) {
            num += (anglar_list[i] * 2 + 1) * 2;
            if (num_occupied <= num) {
                if (num_occupied < num) {
                    ve_anglar_list.push_back(anglar_list[i]);
                }
                break;
            }
        }

        for (++i; i < count; ++i) {
            num += (anglar_list[i] * 2 + 1) * 2;
            ve_anglar_list.push_back(anglar_list[i]);
            if (Z <= num) {
                break;
            }
        }

        return ve_anglar_list;
    } else {
        //for higher than Lu(71)
        Z -= 14; //for skip 4f 

        //orders of 5s and 4d are exchanged//
        const std::vector<int> anglar_list2{/*1s*/0, /*2s*/0, /*2p*/1, /*3s*/0, /*3p*/1, /*4s*/0, /*3d*/2, /*4p*/1,
            /*4d*/2, /*5s*/0, /*5p*/1, /*6s*/0, /*5d*/2, /*6p*/1, /*7s*/0, /*6d*/2 };

        std::vector<int> ve_anglar_list;
        const int num_occupied = Z - ve;
        int num = 0;
        const int64_t count = anglar_list2.size();

        int64_t i = 0;
        for (; i < count; ++i) {
            num += (anglar_list2[i] * 2 + 1) * 2;
            if (num_occupied <= num) {
                if (num_occupied < num) {
                    ve_anglar_list.push_back(anglar_list2[i]);
                }
                break;
            }
        }

        for (++i; i < count; ++i) {
            num += (anglar_list2[i] * 2 + 1) * 2;
            ve_anglar_list.push_back(anglar_list2[i]);
            if (Z <= num) {
                break;
            }
        }

        return ve_anglar_list;

    }
}

