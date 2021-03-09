#include "exception.h"

Quynh::exception::exception(std::string _Message)
{
    this->Message  = _Message;
    this->Errorno = 0;
}

Quynh::exception::exception(std::string _Message, int _Errorno)
{
    this->Message = _Message;
    this->Errorno = _Errorno;
}

void Quynh::exception::print()
{
    std::cout  << "Caution failed  " << this->Message << std::endl;
    if (this->Errorno != 0) std::cout << " the errorno is: " << this->Errorno << std::endl;
}

