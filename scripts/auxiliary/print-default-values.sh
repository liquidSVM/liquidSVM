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


 


####################################################################################
#
# This script prints default values for the standard arguments from the command line
#
####################################################################################


COMMAND=$1
EXTRA_OPS=$2
DISPLAY_PART=$3

echo
echo $COMMAND" is used in the following way:"
echo

if [[ -z $DISPLAY_PART ]]
then
	echo $COMMAND" <base_filename> [<display>] [<threads>] [<partition>] [<grid_size>]" 
	echo "                [<adaptive_grid_search>] [<random_seed>] "$EXTRA_OPS
else
	echo $COMMAND" <base_filename> [<display>] [<threads>] [<grid_size>] [<random_seed>]" 
	echo "                "$EXTRA_OPS
fi


echo 
echo Default values are:
echo
echo display:"               "$DISPLAY
echo threads:"               "$THREADS
if [[ -z $DISPLAY_PART ]]
then
	echo partition:"             "$PARTITION_CHOICE
fi
echo grid size:"             "$GRID_CHOICE
if [[ -z $DISPLAY_PART ]]
then
	echo adaptive_grid_search:"  "$ADAPTIVITY_CONTROL
fi
echo random_seed:"           "$RANDOM_SEED




