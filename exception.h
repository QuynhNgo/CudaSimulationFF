#ifndef EXCEPTION_H
#define EXCEPTION_H
#include<string>
#include<iostream>


namespace Quynh{
class exception
{
public:
    exception(std::string _String);
    exception(std::string _String, int _Index);
    void print();
private:
    std::string Message;
    int Errorno;
};
}
#endif // EXCEPTION_H
