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
		#undef MSVISUAL_LEGACY
		#if _MSC_VER < 1600
			#error For liquidSVM the MS Visual C++ compiler needs to be at least version MSVC++ 10.0!
		#elif  _MSC_VER < 1800
			#define MSVISUAL_LEGACY
		#endif
	#elif __cplusplus < 201103L
		#error For liquidSVM compilers other than GCC require at least C++11 standard activated!
	#endif
#endif



// Now decide if TR1 experimental code needs to be used for medium-aged 
// Versions of GCC or if the C++ standard is too old for unordered maps.
 
#if defined(__GNUC__)
	#if __cplusplus <= 199711L
		#undef USE_TR1_CODE
		#define FALL_BACK_MAP
	#elif __cplusplus < 201103L 
		#define USE_TR1_CODE
		#undef FALL_BACK_MAP
	#else	
		#undef USE_TR1_CODE
		#undef FALL_BACK_MAP
	#endif	
#else
	#undef USE_TR1_CODE
	#if __cplusplus < 201103L
		#define FALL_BACK_MAP
	#else
		#undef FALL_BACK_MAP
	#endif
#endif


	
	
	
#endif
