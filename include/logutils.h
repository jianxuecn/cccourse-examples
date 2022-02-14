#ifndef __logutils_h
#define __logutils_h

#include <iostream>

#define OUTPUT_SOMETHING(os, content) do { os << content; } while (0)

#define PRINT_INFO(x)    OUTPUT_SOMETHING(std::cout, x << std::endl)
#define PRINT_DEBUG(x)   OUTPUT_SOMETHING(std::cout, "|   DEBUG | in " __FILE__ ", line " << __LINE__ << ": "<< x << std::endl)
#define PRINT_ERROR(x)   OUTPUT_SOMETHING(std::cout, "|   ERROR | in " __FILE__ ", line " << __LINE__ << ": "<< x << std::endl)
#define PRINT_WARNING(x) OUTPUT_SOMETHING(std::cout, "| WARNING | in " __FILE__ ", line " << __LINE__ << ": "<< x << std::endl)
#define PRINT_SEGMENT(x) OUTPUT_SOMETHING(std::cout, x)

#define LOG_INFO(x)    PRINT_INFO(x)
#define LOG_DEBUG(x)   PRINT_DEBUG(x)
#define LOG_ERROR(x)   PRINT_ERROR(x)
#define LOG_WARNING(x) PRINT_WARNING(x)
#define LOG_SEGMENT(x) PRINT_SEGMENT(x)

#endif
