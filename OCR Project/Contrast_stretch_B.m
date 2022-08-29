function stretched =Contrast_stretch_B(gray)
    min_P=double(min(gray(:)))%Min intensity=13
    max_P=double(max(gray(:)))%Max intensity=204
    stretched = uint8((double(gray(:,:))-min_P).*(255/(max_P-min_P)));
end