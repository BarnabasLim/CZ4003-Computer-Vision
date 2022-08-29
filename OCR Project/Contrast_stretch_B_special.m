function stretched =Contrast_stretch_B_special(gray)
    P=gray
    P(find(gray==255))=0;
    figure('Name',"10 Problem 1");imshow(P)
    min_P=double(min(P(:)))%Min intensity=13
    max_P=double(max(P(:)))%Max intensity=204
    P=uint8((double(P(:,:))-min_P).*(255/(max_P-min_P)));
    P(find(gray==255))=255;
    figure('Name',"10 Problem 2");imshow(P)
    stretched =P;
end