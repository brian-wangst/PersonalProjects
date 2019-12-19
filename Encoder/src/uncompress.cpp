#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdint.h>

#include "HCNode.hpp"
#include "HCTree.hpp"
#include "BitInputStream.hpp"

using namespace std;

void print_usage(char ** argv) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " INFILE OUTFILE [-b]" << endl;
    cout << "Command-line flags:" << endl;
    cout << "  -b: switch to bitwise mode" << endl;
}

/**
 * Reads infile, constructs a Huffman coding tree based on its header,
 * and produces an uncompressed version in outfile.
 * For debugging purposes, uses ASCII '0' and '1' rather than bitwise I/O.
 */
void uncompressAscii(const string & infile, const string & outfile) {
        vector <int> freqs(256,0);
    char c;


    std::ofstream out;
    std::ifstream in;
    std::ifstream file;
    out.open(outfile, ios::binary);
    in.open(infile,ios::binary);
    string t;
    int count = 0;
    char index;
    
    for(int i = 0; i < 256; ++i){
          if( in >> t){
            freqs.at(i) = std::stoi(t);
          if(std::stoi(t) == 0){
              count ++;
          }
          if(std::stoi(t) != 0){
              index = i;
          }
          }
    }
    
    HCTree tt;
    tt.build(freqs);
    
    if(count == 255){
        for(int j = 0; j < freqs.at(index); ++j)
            out << index;
            return;
    }
    

    
    if((in.get()) == EOF){
        return;
    }
    
    while(!in.eof()){
        if(in.peek() == -1){
            break;
        }
        out << tt.decode(in);
    }
    out.close();
    in.close();
}

/**
 * Reads infile, constructs a Huffman coding tree based on its header,
 * and produces an uncompressed version in outfile.
 * Uses bitwise I/O.
 */
void uncompressBitwise(const string & infile, const string & outfile) {
        vector <int> freqs(256,0);
    int c;
    std::ofstream out;
    std::ifstream in;
    std::ifstream file;
    out.open(outfile, ios::binary);
    in.open(infile,ios::binary);
    int count = 0;
    int total = 0;
    char index;
    
    for(int i = 0; i < 256; ++i){
         in.read((char*)&c, sizeof(c));
            freqs.at(i) = c;
            total = c + total;
            
          if(c == 0){
              count ++;
          }
          if( c != 0){
              index = i;
          }
    }
    
    HCTree tt;
    tt.build(freqs);
    
    if(total == 0){
        return;
    }
    
    if(count == 255){
        for(int j = 0; j < freqs.at(index); ++j){
            out << index;
            return;
    }
    }
    

    if(in.peek() == -1){

        return;
    }
    
    BitInputStream bitIn = BitInputStream(in);
    while(total !=0 ){
        out << tt.decode(bitIn);
        total--;
    }
    // while(!in.eof()){
    //     if(in.peek() == -1){
    //         break;
    //     }
    //     out << tt.decode(bitIn);
    //     total = total - 1;
    //     if(total == 0){
    //         break;
    //     }
    // }
    out.close();
    in.close();
}

int main(int argc, char ** argv) {
    string infile = "";
    string outfile = "";
    bool bitwise = false;
    for (int i = 1; i < argc; i++) {
        string currentArg = argv[i];
        if (currentArg == "-b") {
            bitwise = true;
        } else if (infile == "") {
            infile = currentArg;
        } else {
            outfile = currentArg;
        }
    }

    if (infile == "" || outfile == "") {
        cout << "ERROR: Must provide input and output files" << endl;
        print_usage(argv);
        return 1;
    }

    if (bitwise) {
        uncompressBitwise(infile, outfile);
    } else {
        uncompressAscii(infile, outfile);
    }

    return 0;
}
