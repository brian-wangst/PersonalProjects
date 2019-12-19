#ifndef BITOUTPUTSTREAM_HPP
#define BITOUTPUTSTREAM_HPP

#include <iostream>

using namespace std;

class BitOutputStream {
private:
    ostream & out;
    int nbits;
    unsigned char buff;
    int freq;

    

public:
    BitOutputStream(ostream & o);
    void writeBit(bool bit);
    void flush();
    int getBits();
};

#endif // BITOUTPUTSTREAM_HPP
