#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <ostream>
#include <chrono>


#include "Graph.hpp"
#include "Node.hpp"


using namespace std;

void usage(char* program_name) {
  cerr << program_name << " called with incorrect arguments." << endl;
  cerr << "Usage: " << program_name
       << " friendship_pairs_file k_value output_file"
       << endl;
  exit(-1);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    usage(argv[0]);
  }
  
  
  char* graph_filename = argv[1];
  char* output_filename = argv[3];
  istringstream ss(argv[2]);
  
  Graph test;
  test.loadFromFile(graph_filename);
  
  ofstream out;

  out.open(output_filename, ios::binary);
  



  
  int k;
  ss >> k;
  


  vector<pair<string,int>> core;
  string output;
  
  output = test.socialgathering(core, k);
  
  out << output;
  
}