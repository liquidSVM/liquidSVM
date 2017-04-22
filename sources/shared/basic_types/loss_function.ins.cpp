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




inline double sign(double x)
{
	return ( (x >= 0.0)? 1.0:-1.0);
}

//**********************************************************************************************************************************


inline double classification_loss(double y, double t)
{
	return ( (y * sign(t) <= 0.0)? 1.0:0.0);
}


//**********************************************************************************************************************************


inline double neg_classification_loss(double y, double t)
{
	return ( (y <= 0.0)? classification_loss(y,t):0.0);
}


//**********************************************************************************************************************************


inline double pos_classification_loss(double y, double t)
{
	return ( (y >= 0.0)? classification_loss(y,t):0.0);
}
