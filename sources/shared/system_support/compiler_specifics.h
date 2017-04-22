// Copyright 2015, 2016, 2017 Ingo Steinwart
//
// This file is part of liquidSVM.
//
// liquidSVM is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as 
// published by the Free Software Foundation, either version 3 of the 
// License, or (at your option) any later version.
//
// liquidSVM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.


#if !defined (COMPILER_SPECIFICS_H) 
 #define COMPILER_SPECIFICS_H


 
//  First check if C++11 is activated for compilers other than GCC
 
#if !defined(__GNUC__) || defined(__clang__)
	#if defined(_MSC_VER) 
		#if _MSC_VER < 1800
			#error For liquidSVM the MS Visual C++ compiler needs to be at least version MSVC++ 12.0!
		#endif
	#elif __cplusplus < 201103L
		#error For liquidSVM compilers other than GCC require at least C++11 standard activated!
	#endif
#endif



// Now decide if TR1 experimental code needs to be used for old 
// Versions of GCC. Note that for stone-aged versions of GCC there
// is no TR1 code available, and hence subsequent error messages will
// occur.
 
#if defined(__GNUC__) && __cplusplus < 201103L 
	#define USE_TR1_CODE
#else
	#undef USE_TR1_CODE
#endif
	
	
	
#endif
