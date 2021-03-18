#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<vector>


/*
 * @des Utility class 
 */

class utils
{
public:
    static int stringtolong(std::string);
    static std::vector<std::string> tokenize(std::string, char);
    static double stringtoDouble(std::string);
	// Print each matrix in one row of a file
	/*
	 * @des print an array into file with NumbCoactivationMatrices rows
	 */
	static void printToFile(float *List, int NumberOfEntriesInOneRow, int TotalNumbEntries, std::string filename);
private:
    utils();
};

#endif // UTILS_H
