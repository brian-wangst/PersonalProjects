#include "BitInputStream.hpp"

// TODO (final)
BitInputStream::BitInputStream(istream & i) : in(i), nbits(8){}

bool BitInputStream::readBit() {        //read a single bit from buff
    if(nbits == 8){
        fill();
    }
    unsigned char temp = buf;
    if((temp & (1 << (7 - nbits))) != 0){
        nbits = nbits + 1;
        return true;
    }
    
    nbits = nbits + 1;
    return false;  // TODO (final)
}


void BitInputStream:: fill(){        //helper method for readBit()
    buf = in.get();
    nbits = 0;
}