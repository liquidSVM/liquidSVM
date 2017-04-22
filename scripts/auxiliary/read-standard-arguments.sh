#!/bin/bash
 
# Copyright 2015, 2016, 2017 Ingo Steinwart
#
# This file is part of liquidSVM.
#
# liquidSVM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as 
# published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# liquidSVM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.


 


############################################################
#
# This script reads standard arguments from the command line
#
############################################################



BASE_FILENAME="$1"
if ! [[ -z "$2" ]]
then
	DISPLAY="$2"
fi

if ! [[ -z "$3" ]]
then
	THREADS="$3"
fi

if ! [[ -z "$4" ]]
then
	PARTITION_CHOICE="$4"
fi

if ! [[ -z "$5" ]]
then
	GRID_CHOICE="$5"
fi

if ! [[ -z "$6" ]]
then
	ADAPTIVITY_CONTROL="$6"
fi

if ! [[ -z "$7" ]]
then
	RANDOM_SEED="$7"
fi
