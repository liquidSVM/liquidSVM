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





//**********************************************************************************************************************************


template <typename Template_type> bool Ttrain_val_info::apply_mask_entry(Template_type& entry, Template_type mask)
{
	if (mask < 0.0)
		entry = mask;
	else if (entry > mask)
	{
		entry = Template_type(IGNORE_VALUE);
		return true;
	}
	return false;
}


