#include "BitOutputStream.hpp"

// TODO (final)
BitOutputStream::BitOutputStream(ostream & o) : out(o), buff(0), nbits(0), freq(0) {}

void BitOutputStream::writeBit(bool bit) {      //use nbits to keep track where i am in the byte
    if(nbits == 8){
        flush();
    }
    if(bit == false){        // if bit == 0, then already 0 at that place and add 1 to bits
        nbits = nbits + 1;
    }
    
    else{
        buff = buff | (1 << (7 - nbits));     // create a mask of 1 index position and add to original buffer.
        nbits = nbits + 1;
    }
}

void BitOutputStream::flush() {
    out.put(buff);
    out.flush();     //not sure about this part? stepick said optional because slower, where to put instead?
    buff = 0;
    nbits = 0;
}
