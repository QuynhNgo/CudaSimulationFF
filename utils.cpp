#include "utils.h"
#include "exception.h"

std::vector<std::string> utils::tokenize(std::string _Token, char _Separator)
{
    std::vector<std::string> _Tokens;
    std::string _Qq;
    std::stringstream ss(_Token);
    while(std::getline(ss, _Qq, _Separator))
    {
        _Tokens.push_back(_Qq);
    }
    return _Tokens;
}



int utils::stringtolong(std::string _Token)
{
    int _Val;
    errno = 0;

    _Val = atoi(_Token.data());
    //val = strtol(token.c_str(), &Endptr, 10);

    if((errno ==ERANGE&&(_Val==LONG_MAX||_Val ==LONG_MIN))||(errno != 0 && _Val ==0))
    {
        throw Quynh::exception("String to long int failed", errno);
    }
    //if (Endptr == token.c_str()){
       // std::cout<<"token was"<<token<<std::endl;
      //  throw Exception(" String to long int failed, no digit were found!",errno);}
    return _Val;
}

double utils::stringtoDouble(std::string _Token)
{
    char *_Endptr;
    double val;
    errno = 0; /* To distinguish success/failure after call */
    val = strtod(_Token.c_str(), &_Endptr);
    /* Check for various possible errors */
    if (errno != 0 && val == 0) {
        throw Quynh::exception("stringToDouble failed", errno);
    }
    if (_Endptr == _Token.c_str()) {
        throw Quynh::exception("stringToDouble failed: No digits were found", errno);
    }
    return val;
}
