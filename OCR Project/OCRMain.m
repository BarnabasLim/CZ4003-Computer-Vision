clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Readme%
%Please Select:
    %1.Sample_no-> indicate which sample to run the preproceesing on
    %2.preprocessing_steps-> Uncomment the preprocessing steps to be execused
    %their corresponding report section is indicated.
%%%%%%%%% 1.Select Sample%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sample_no=2
%%%%%%%%% 2.Uncomment Processing Steps to be executed%%%%%%%%%%%%%%
%"0 Grey", "1 OTSU", "2 Deskew" "3 Homomorphic Filtering", "4 Histogram Equilisation",...
%        "5 Adaptive Thresholding", "6 Opening","7 erode",...
%        "8 Gaussian Blurring/Filtering","9 imtophat"

%4
%OTSU
%preprocessing_steps=[1]

%5.4.1
%Deskew-> Homo-> OTSU
%%%%%%%%%%%%%%%%%%%%%%%BEST for sample 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%preprocessing_steps=[2 3 1]
%Deskew-> Homo->Hist Equi ->OTSU
%preprocessing_steps=[2 3 4 1]

%5.4.2
%Deskew-> Gaussian ->Adapt_T-> erode
%%%%%%%%%%%%%%%%%%BEST for in General Sample 1 and Sample 2%%%%%%%%%%%%%%%
%preprocessing_steps=[2 8 5 7]

%5.4.3
%Deskew->Hist Equi-> Tophat-> erode-> Adapt_T
%preprocessing_steps=[2 4 9 7 5]

%6
%Deskew-> Gaussian ->Adapt_T-> erode->10->Homo+ Segment-> OTSU   
%%%%%%%%%%%%%%%%%%%%%BEST for sample 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
preprocessing_steps=[2 8 5 7 10 3 1]

%Others
%Deskew-> Gaussian ->Adapt_T-> erode->10->Tophat+ Segment-> erode-> Adapt_T
%preprocessing_steps=[2 8 5 7 10 9 7 5]
%Deskew->Hist Equi-> Tophat->Gaussian ->Adapt_T-> erode
%preprocessing_steps=[2 4 9 8 5 7]
%Deskew-> Homo->Hist Equi-> Gaussian ->Adapt_T-> erode
%preprocessing_steps=[2 3 8 5 7]

%add the current folder to the Python search path.
%Run Matlab 2020
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

tool_Name={"Grey", "OTSU", "Deskew" "Homomorphic Filtering", "Histogram Equilisation",...
        "Adaptive Thresholding", "Opening","Erode",...
        "Gaussian Blurring/Filtering","imtophat","10"}

img_file=["resource/sample01.png " "resource/sample02.png "]

N=size(preprocessing_steps,2)
img_raw_results_cell=cell(N+1,1);
Results_overall=zeros(2,N+1)
Pc = imread(img_file(sample_no));

whos Pc
if(size(Pc,3)==3)
    P_gray = rgb2gray(Pc);
else
    P_gray = Pc;
end
img_raw_results_cell{1}=P_gray
mask_array=-1
for i=1:N
    %1 OTSU 
    if preprocessing_steps(i)==1
        %2 OTSUP=double(py.Overall_OCR.deskew(P));
        P=Contrast_stretch_B(img_raw_results_cell{i});      %2.1 Contrast stretching 
        t=OTSU_B(P,true);                                   %2.2 OTSU Global Thesholding
        P=P>t;
        img_raw_results_cell{i+1}=P;
    
    %Step 2: Deskew 
    %2 Deskew
    elseif preprocessing_steps(i)==2
        img_raw_results_cell{i+1}=Contrast_stretch_B(double(py.numpy.array(py.Overall_OCR.deskew(img_raw_results_cell{i}))));
    
    
    %Step 4: Binarisation
    %3 Homomorphic Filtering
    elseif preprocessing_steps(i)==3
        %3 Homomorphic Filtering + Contrast Stretching +OTSU
        P=HOMO_Filtering_B(img_raw_results_cell{i});       %3.1 Homomorphic Filtering
        img_raw_results_cell{i+1} = Contrast_stretch_B(P); %3.2 Contrast stretching 
        if mask_array~=-1
            img_raw_results_cell{i+1}(mask_array) = 255;
            img_raw_results_cell{i+1} = Contrast_stretch_B_special(img_raw_results_cell{i+1});
        end
        
    %4 Histogram Equilisation
    elseif preprocessing_steps(i)==4
        P=Contrast_stretch_B(img_raw_results_cell{i}); 
        img_raw_results_cell{i+1} = histeq(P ,255); %histogram equilisation
     
    %5 Adaptive Thresholding
    elseif preprocessing_steps(i)==5
        %S_varry=linspace(0.659,0.67,50)
        S_varry=linspace(0.3,0.9,50);
        %S_varry=linspace(0,1,100)
        result_to=zeros(1,size(S_varry,2));
        for k=1:size(S_varry,2)
            BW=BW_adaptT(img_raw_results_cell{i}, S_varry(k)); 
            result_pre=cellfun(@double,cell(py.Overall_OCR.tesseractOCR(BW,sample_no)));
            result_to(k)=result_pre(2);
        end
        [M,I] = min(result_to);
        M
        S_varry(I)%0.661
        figure;plot(S_varry, result_to);
        hold on;
        plot(S_varry(I),M,'o',...
            'MarkerEdgeColor','red',...
            'MarkerFaceColor',[1 .6 .6])
        ;xlabel('Sensitivity [0.659,0.67]');ylabel('Accuracy');
        legend(['Levenshtein Dist: ']...
        ,[strcat('Best Levenshtein Dist: @(S=',num2str(round(S_varry(I),4)),')=',num2str(round(M,2)))] ...
        ,'Location','best');
        result_O=cellfun(@double,cell(py.Overall_OCR.tesseractOCR(img_raw_results_cell{i},sample_no)));
        if result_O(2)<M
            img_raw_results_cell{i+1}=img_raw_results_cell{i}
        else
            img_raw_results_cell{i+1}=BW_adaptT(img_raw_results_cell{i}, S_varry(I)); 
        end

    %Step 5: Skeletionisation
    %6 Opening, erode,7,erode,9,imtophat 
    elseif ismember(preprocessing_steps(i),[6,7,9])
        shape_element={'diamond','disk','square';}
        shape_element_no=linspace(1,size(shape_element,2),size(shape_element,2));
        if preprocessing_steps(i)==6
            r=[3,4,5,6]
        elseif preprocessing_steps(i)==7
            r=[3,4,5,6]
        elseif preprocessing_steps(i)==9
            r=[9,10,11,12]
        end
        pic=imcomplement(img_raw_results_cell{i});
        result_to= zeros(size(r,2),size(shape_element,2))
        [X,Y] = meshgrid(shape_element_no,r)
        for j = 1:size(shape_element,2)
            for p = 1:size(r,2)
                se = strel(shape_element{j},r(p));
                if preprocessing_steps(i)==6
                    img = imcomplement(imopen(pic,se)); 
                elseif preprocessing_steps(i)==7
                    img = imcomplement(imerode(pic,se));
                elseif preprocessing_steps(i)==9
                    img = imtophat(imcomplement(pic),se);
                end
                result_pre=cellfun(@double,cell(py.Overall_OCR.tesseractOCR(img,sample_no)));
                result_to(p,j)=result_pre(2);
            end
        end
        minMatrix = min(result_to(:));
        [row,col] = find(result_to==minMatrix);
        figure( 'Position', [10 10 900 600]);
        subplot(3,2,1);mesh(X,Y,result_to,'FaceAlpha','0.8'),title('imopen()'),xlabel('shape_element'),ylabel('r'),zlabel('result');
        result_to
        se = strel(shape_element{col},r(row));
        result_O=cellfun(@double,cell(py.Overall_OCR.tesseractOCR(img_raw_results_cell{i},sample_no)));
        if result_O(2)<minMatrix
            img_raw_results_cell{i+1}=img_raw_results_cell{i}
        else
            if preprocessing_steps(i)==6
                    img_raw_results_cell{i+1} = imcomplement(imopen(imcomplement(img_raw_results_cell{i}),se));
            elseif preprocessing_steps(i)==7
                    img_raw_results_cell{i+1} = imcomplement(imerode(imcomplement(img_raw_results_cell{i}),se));
            elseif preprocessing_steps(i)==9
                    img_raw_results_cell{i+1} = imtophat(img_raw_results_cell{i},se);
                    if mask_array~=-1
                        img_raw_results_cell{i+1}(mask_array) = 255;
                        img_raw_results_cell{i+1} = Contrast_stretch_B_special(img_raw_results_cell{i+1});
                    end
            end
        end

    %8 Gaussian Blurring/Filtering
    elseif preprocessing_steps(i)==8
        sigma=linspace(0.5,4,10);
        result_to=zeros(1,size(sigma,2));
        for k=1:size(sigma,2)
            img=imgaussfilt(img_raw_results_cell{i},sigma(k));
            result_pre=cellfun(@double,cell(py.Overall_OCR.tesseractOCR(img,sample_no)));
            result_to(k)=result_pre(2);
        end
        [Sigma_M,I] = min(result_to);
        Sigma_M
        sigma(I)%0.661
        img_raw_results_cell{i+1}=imgaussfilt(img_raw_results_cell{i},sigma(I))
    %10 Mask From Best Soltuion method
    elseif preprocessing_steps(i)==10
        A=find(preprocessing_steps==5);
        pic=img_raw_results_cell{A+1};
        pic_invert=(pic==0);
        se = strel('disk',3);
        dilatedI = imdilate(pic_invert,se);
        figure;imshowpair(pic_invert,dilatedI,'montage')
        
        prev_img=img_raw_results_cell{i}
        mask_array=find(dilatedI==0);
        
        prev_img(find(dilatedI==0)) = 255;
        img_raw_results_cell{i+1}=img_raw_results_cell{2}
    end

end
figure( 'Position', [5 5 1400 500]);
for i=1:size(img_raw_results_cell,1)
    Results_overall(:,i)=cellfun(@double,cell(py.Overall_OCR.tesseractOCR(img_raw_results_cell{i},sample_no)));
    if i==1;
        Process_Name=tool_Name{i};
    else
        Process_Name=tool_Name{preprocessing_steps(i-1)+1};
    end
    subplot(1,N+1,i);imshow(img_raw_results_cell{i});title([strcat(Process_Name,'\rightarrow') ,strcat('Levenshtein Dist: ', num2str(round(Results_overall(2,i),2)))]);
end

figure;imshowpair(img_raw_results_cell{1},img_raw_results_cell{N+1},'montage'),title([strcat('Orignal Accuracy: ',num2str(round(Results_overall(1,1),2)),'% Levenshtein Dist:  ', num2str(round(Results_overall(2,1),2))...
    , '. Vs .', tool_Name{preprocessing_steps(N)+1},num2str(round(Results_overall(1,N+1),2)),'% Levenshtein Dist: ', num2str(round(Results_overall(2,N+1),2)))]);

S=4
figure;imshowpair(img_raw_results_cell{1},img_raw_results_cell{S},'montage'),title([strcat('Orignal Accuracy: ',num2str(round(Results_overall(1,1),2)),'% Levenshtein Dist:  ', num2str(round(Results_overall(2,1),2))...
    ,'. Vs .', tool_Name{preprocessing_steps(S-1)+1},num2str(round(Results_overall(1,S),2)),'% Levenshtein Dist: ', num2str(round(Results_overall(2,S),2)))]);
