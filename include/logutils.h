/**
 * -------------------------------------------------------------------------------
 * This source file is part of cccourse-examples, one of the examples for
 * the Cloud Computing Course and the Computer Graphics Course at the School
 * of Engineering Science (SES), University of Chinese Academy of Sciences (UCAS).
 * Copyright (C) 2022 Xue Jian (xuejian@ucas.ac.cn)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------------
 */
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
