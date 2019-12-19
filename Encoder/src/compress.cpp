#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

#include "HCNode.hpp"
#include "HCTree.hpp"
#include "BitOutputStream.hpp"

using namespace std;

void print_usage(char ** argv) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " INFILE OUTFILE [-b]" << endl;
    cout << "Command-line flags:" << endl;
    cout << "  -b: switch to bitwise mode" << endl;
}

/**
 * Reads infile, constructs a Huffman coding tree based on its contents,
 * and produces a compressed version in outfile.
 * For debugging purposes, uses ASCII '0' and '1' rather than bitwise I/O.
 */
void compressAscii(const string & infile, const string & outfile) {
    vector <int> freqs(256,0);
    int c;

    std::ofstream out;
    std::ifstream in;
    out.open(outfile, ios::binary);
    in.open(infile,ios::binary);
 
    while((c = in.get()) != EOF){ 
        freqs.at(c) = freqs.at(c) + 1;
    }
    for(int i = 0; i < freqs.size(); ++i){
        out << freqs.at(i) << endl;
    }
    HCTree t;
    t.build(freqs);
    
    in.clear();
    in.seekg(0,ios::beg);
    
    
  
    while((c = in.get()) != EOF){
        t.encode(c,out);
    }
   
  in.close();
  out.close();
}

/**
 * Reads infile, constructs a Huffman coding tree based on its contents,
 * and produces a compressed version in outfile.
 * Uses bitwise I/O.
 */
void compressBitwise(const string & infile, const string & outfile) {
    vector <int> freqs(256,0);
    int c;
    int filesize = 0;
    std::ofstream out;
    std::ifstream in;
    out.open(outfile, ios::binary);
    in.open(infile,ios::binary);
    int count = 0;
 
    while((c = in.get()) != EOF){ 
        freqs.at(c) = freqs.at(c) + 1;
        count ++;
    }
    if(count == 0){
        return;
    }
    
    for(int i = 0 ; i < freqs.size(); ++i){
        out.write((char*)&freqs.at(i), sizeof(freqs.at(i)));
    }

    
     in.clear();
     in.seekg(0,ios::beg);
    unsigned char k;
    HCTree t;
    t.build(freqs);
    BitOutputStream bitout = BitOutputStream(out);
    while((k = in.get()) != EOF){
        if(in.peek() == -1){
            t.encode(k,bitout);
            break;
        }
            t.encode(k,bitout);
        }
        bitout.flush();
        
        in.close();
        out.close();
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
        compressBitwise(infile, outfile);
    }
    // } else {
    //     compressAscii(infile, outfile);
    // }

    return 0;
}
