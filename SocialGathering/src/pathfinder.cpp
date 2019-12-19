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
using namespace std::chrono;

void usage(char* program_name) {
  cerr << program_name << " called with incorrect arguments." << "\n";
  cerr << "Usage: " << program_name
       << " friendship_pairs_file test_pairs_file output_file"
       << "\n";
  exit(-1);
}

int main(int argc, char* argv[]) {
  
  if (argc != 4) {
    usage(argv[0]);
  }
  

  char* graph_filename = argv[1];
  char* pairs_filename = argv[2];
  char* output_filename = argv[3];

  //TODO   


  Graph test;
  test.loadFromFile(graph_filename);
  
  ifstream infile(pairs_filename);
  ofstream out;

  out.open(output_filename, ios::binary);

 


  string fullOutput = "";
  while (infile) {

    string s;
    if (!getline(infile, s)) break;

    istringstream ss(s);
    vector<string> record;
  
    while (ss) {
      string s;
      if (!getline(ss, s, ' ')) break;
      record.push_back(s);  
    }


    if (record.size() != 2) {
      continue;
    }
    
    if((test.map[record.at(0)].size() == 0) || (test.map[record.at(1)].size() == 0)){     //if either node doesn't exist, don't step into function
      fullOutput = fullOutput + "\n";
      continue;
    }

    string output = test.pathfinder(test.map[record.at(0)].at(0),test.map[record.at(1)].at(0));   //call pathfinder function
    
    if(output == "\n"){
      fullOutput = fullOutput + "\n";
      continue;
    }
    
    fullOutput = fullOutput + output + "\n";

    
  
 /* You can call the pathfinder function from here */
}
  out << fullOutput;
  out.close();
  infile.close();


}  