% Copyright 2015-2017 Philipp Thomann
% 
% This file is part of liquidSVM.
% 
% liquidSVM is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as
% published by the Free Software Foundation, either version 3 of the
% License, or (at your option) any later version.
% 
% liquidSVM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU Affero General Public License for more details.
% 
% You should have received a copy of the GNU Affero General Public License
% along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.

function makeliquidSVM(varargin)
% makeliquidSVM compile liquidSVM library
% The first argumment can be one of native, generic, debug, or empty.
% These are different compile optimization strategies.
% Further arguments are passed as compiler flags:
%   makeliquidSVM generic -mavx
%   makeliquidSVM -mavx

[theDir,~] = fileparts(mfilename('fullpath'));
addpath(fullfile(theDir,'classes'));
addpath(fullfile(theDir,'functions'));

makeDoc = false;
if makeDoc
    liquidSVM_makeDocumentation();
end

target='native';

if nargin > 0 && any(strcmp(varargin{1},{'native','generic','debug','empty'}))
    target = varargin{1};
    varargin = varargin(2:end);
end

disp( [ 'Using target: ', target ] )



cppfile = fullfile(theDir,'mexliquidSVM.cpp');
otherargs =  {'-v'};

switch target
    case 'native'
        CXX_FLAGS='-march=native -O3';
    case 'generic'
        CXX_FLAGS='-mtune=generic -msse2 -O3';
    case 'debug'
        CXX_FLAGS='-mtune=generic -msse2 -g';
    case 'empty'
        CXX_FLAGS='';
end

SPECIFICS = {};

CXX_FLAGS = strsplit([ CXX_FLAGS, ' -std=c++0x']);
CXX_FLAGS = [ CXX_FLAGS; varargin ];
TOOLBOX_INCLUDE_PATH = [ '-I"' fullfile(theDir,'src') '"'];
ipath = {  '-I../..' '-I../../bindings'  '-IC:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\src\includes' TOOLBOX_INCLUDE_PATH };
disp( [ 'Using CXX_FLAGS: ', strjoin(CXX_FLAGS, ' ') ] )
disp( [ 'Using INCLUDE_FLAGS: ', strjoin(ipath, ' ') ] )

if exist('OCTAVE_VERSION', 'builtin') == 0
    % fild matlab root path and include external library of minGW if
    % available
    mRoot = matlabroot;
    minGwLibFolder = fullfile(mRoot,'extern','lib','win64','mingw64');
    if exist(minGwLibFolder,'dir');
%        SPECIFICS = {'-lut','-largeArrayDims','-DDO_MATLAB'};
        SPECIFICS = {['-L',minGwLibFolder],'-lut','-largeArrayDims','-DDO_MATLAB'}; 
    else
        % let's do MATLAB
        SPECIFICS = {'-lut','-largeArrayDims','-DDO_MATLAB'};        
    end
    % For MATLAB the compiler flags need to be in one argument!
    CXX_FLAGS = { [ 'CXXFLAGS=$CXXFLAGS ', strjoin(CXX_FLAGS, ' ') ] };
    otherargs = [otherargs, {'-outdir',theDir}];
else
	% let's do Octave
	SPECIFICS = {'-DDO_OCTAVE'};
end

a = [otherargs, CXX_FLAGS, ipath, SPECIFICS, {cppfile} ];

disp( [ 'Using mex ', strjoin(a, ' ') ] )

mex(a{:});
