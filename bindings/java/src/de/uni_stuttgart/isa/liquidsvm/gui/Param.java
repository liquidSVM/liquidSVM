// Copyright 2015-2017 Philipp Thomann
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


package de.uni_stuttgart.isa.liquidsvm.gui;

public class Param {
	public Param(String name, String type, int optional) {
		super();
		this.name = name;
		this.type = type;
		this.optional = optional;
	}
	public String name;
	public String type;
	public int optional;
	
	@Override
	public String toString() {
		return "Param [name=" + name + ", type=" + type + ", optional="
				+ optional + "]";
	}
	
}
