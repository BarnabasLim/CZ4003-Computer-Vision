function BW =BW_adaptT(gray, sensitivity)
%https://www.mathworks.com/help/images/ref/adaptthresh.html#namevaluepairs
%Sensitivity: Determine which pixels get thresholded as foreground pixels, 
%specified as a number in the range [0, 1]. 
%High sensitivity values lead to thresholding more pixels as foreground, 
%at the risk of including some background pixels.
I=gray;

T = adaptthresh(I, sensitivity);
BW = (imbinarize(I,T));
%BW=double((I>100).*255);
%figure
%imshowpair(I, BW, 'montage')
end