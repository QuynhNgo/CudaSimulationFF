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
private:
    utils();
};

#endif // UTILS_H
