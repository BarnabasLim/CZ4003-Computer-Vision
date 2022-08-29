function map = disparity_map_Barn(Pl, Pr, tempx,tempy)

[h, w] = size(Pl);
% Empty Map
map = ones(h, w);

%Half the size of template
half_tx=floor(tempx/2);
half_ty=floor(tempy/2);

max_disparity=15;

for row = half_tx+1:h-half_tx
    for xl = half_ty+1:w-half_ty
        %Template
        T = Pl(row-half_ty:row+half_ty,xl-half_tx:xl+half_tx);
        %Since right point image will be more left than left point image
        left = xl-max_disparity;
        right = xl;
        if left<half_tx+1
            left = half_tx+1;
        end
        ssd_min = Inf;
        xr_min = right;
        for xr = left:right
            I = Pr(row-half_ty:row+half_ty,xr-half_tx:xr+half_tx);
            ssd=sum(double(I).*double(I),"All")-2*sum(double(I).*double(T),"All");
            if ssd<ssd_min
                ssd_min=ssd;
                xr_min = xr;
            end
        end
        d = xl - xr_min;
        map(row, xl) = -d;
    end
end
map=map(half_tx+1:h-half_tx, half_ty+1:w-half_ty);
end 