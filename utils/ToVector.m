function v = ToVector(im)
% takes MxNx3 picture and returns (MN)x3 vector
sz = size(im);
v = reshape(im, [prod(sz(1:3)) sz(4)]);