function [ flatStruct ] = flattenStructArray( theStruct )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

fnames = fieldnames(theStruct);
flatStruct = theStruct(1);  %will overwrite these fields anyway

for i = 1:length(fnames)
    
    if(ischar(theStruct(1).(fnames{i})))
        flatStruct.(fnames{i}) = {theStruct.(fnames{i})};
    else
        flatStruct.(fnames{i}) = [theStruct.(fnames{i})];
    end

end

