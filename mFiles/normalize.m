function [output] = normalize(input)
    [rows cols] = size(input);
    squaredInput = input.*input;
    sumSquaredInput = sum(sum(squaredInput));
    normVal = sqrt(sumSquaredInput);
    output = input/normVal;
end
