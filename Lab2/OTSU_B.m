function threshold =OTSU_B(gray, Analysis)
%OTSU Thresholding
%Minimise intraclass variance
[count,bins]=imhist(uint8(gray),256);
% size(count)
% intra_class_var=zeros(size(bins,1),1)
% for n=1:size(bins,1)
%     %Mean of L(backgnd)and H(foregnd)
%     mean_L=dot(count(1:n),bins(1:n))/sum(count(1:n));
%     mean_H=dot(count(n+1:256),bins(n+1:256))/sum(count(n+1:256));
%     var_L=dot((bins(1:n)-mean_L).^2,count(1:n)./sum(count));
%     var_H=dot((bins(n+1:256)-mean_H).^2,count(n+1:256)./sum(count));
%     intra_class_var(n)=var_L+var_H
% end
% [M,I] = min(intra_class_var)
% figure;
% plot(bins,intra_class_var);
% hold on;
% plot(bins(I),intra_class_var(I),'o',...
%     'MarkerEdgeColor','red',...
%     'MarkerFaceColor',[1 .6 .6])
% title({'Intra class variance','{\sigma _w}^2={q}_L{\sigma _L}^2+{q}_H{\sigma _H}^2'})
% xlabel('Threshold, t');
% ylabel('Intra class variance, {\sigma _w}^2');


%Maximise intraclass variance
size(count)
inter_class_var=zeros(size(bins,1),1);
for n=1:size(bins,1)
    %Mean of L(backgnd)and H(foregnd)
    mean_L=dot(count(1:n),bins(1:n))/sum(count(1:n));
    mean_H=dot(count(n+1:256),bins(n+1:256))/sum(count(n+1:256));
    weight_L=sum(count(1:n))/sum(count);
    weight_H=sum(count(n+1:256))/sum(count);
    inter_class_var(n)=weight_L*weight_H*(mean_L-mean_H)^2;
end
[M,I] = max(inter_class_var)
%Analysis Report
    if Analysis ==true
        %Plot Interclass variance
        figure( 'Position', [10 10 900 600]);
        
        subplot(1,2,1);plot(bins,inter_class_var);
        hold on;
        plot(bins(I),inter_class_var(I),'o',...
            'MarkerEdgeColor','red',...
            'MarkerFaceColor',[1 .6 .6])
        title({'Inter class variance','{\sigma}^2-{\sigma _w}^2={W}_L{W}_H({\mu _L}+{\mu _H})^2'})
        xlabel('Threshold, t');
        ylabel('Inter class variance, {\sigma}^2-{\sigma _w}^2');

        %Plot histogram with OTSU threshold
        
        subplot(1,2,2);imhist(uint8(gray),256);title("Histogram with OTSU Threshold");
        hold on;
        xline(bins(I),'-r',{'OTSU Threshold',strcat('t= ',num2str(bins(I)))})
    end 

threshold=bins(I)
end
