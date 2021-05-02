clear; clc; close all;
%% Params
% User Params
perc_rate = 0.01;    %Egitim verisinden seçilecek verinin orani % default 0.05

%% Load Images
hyperdata = load('indian_pines.mat');
% hyperdata = hyperdata.salinas;
hyperdata = hyperdata.indian_pines;
%hyperdata = hyperdata.paviaU; 
hyperdata_gt = load('indian_pines_gt.mat');
% hyperdata_gt = hyperdata_gt.salinas_gt; 
hyperdata_gt = hyperdata_gt.indian_pines_gt;
%hyperdata_gt = hyperdata_gt.paviaU_gt;
%% Normalization
%hyperdata = ( hyperdata - min(min(min(hyperdata))) ) /  ( max(max(max(hyperdata))) - min(min(min(hyperdata))) );
[h,w,spec_size] = size(hyperdata);
% data2vector
hypervector = reshape(hyperdata , [h*w,spec_size]);

%% 	Variables

Mask = zeros(size(hyperdata_gt));       % Egitime girecek veri
NumOfClass = max(hyperdata_gt(:));      % Hyper imge'deki class sayisi
NumOfClassElements = zeros(1,NumOfClass);        % Her class'a ait etiket sayisini tutacak degisken
%% Calculate each class's elements number
for i = 1 : 1 : NumOfClass
    NumOfClassElements(i) = double(sum(sum((hyperdata_gt == i)))); %kaç 1 kaç 2 .... kaç 9 var
end
%% ** %Perc_Rate data will select for training vector ** 
for i = 1 : 1 : NumOfClass
    perc_5 = floor(NumOfClassElements(i) * perc_rate);
    [row,col] = find(hyperdata_gt == i);
    %Maskedeki ilgili etiket degeri yüzde 5'e ulasmadigi sürece örnek al
    %maský güncelliyo hyperdata_gt verileri ile gibi düþünebiliriz
    while(sum(sum(Mask == i)) ~= perc_5) 
        x = floor((rand() * (NumOfClassElements(i) - 1)) + 1);
        Mask(row(x),col(x)) = hyperdata_gt(row(x),col(x));
    end
end

%% Class's Nums

ClassSayisi = 0;
for i = 1 : 1 : NumOfClass
   ClassSayisi = ClassSayisi + sum(sum(Mask == i)); 
    %perc deðerine göre aldýn ya onun içinde kaç tane 1 2...9 var
    %mask içinde 1 den kaç tane var + 2 den kaç tane var......9 dan kaç
    %tane var 
end

%% Created Train and Label Vector %%etiket ve eðitimmm miii

[trainingData_row,trainingData_col,values] = find(Mask ~= 0);
%zaten mask içinde 0 olmayan deðerlerin row,col ve etiketlerini tutuyo  
trainingVector = zeros(ClassSayisi,spec_size); %hepsi için
trainingVectorLabel = zeros(ClassSayisi,1); %sýnýflaru tutucak

% ****  Training Vector & Training Vector Label  ****

for i = 1 : ClassSayisi
    trainingVector(i,:)      = hyperdata(trainingData_row(i),trainingData_col(i),:);
    trainingVectorLabel(i,1) = hyperdata_gt(trainingData_row(i),trainingData_col(i));
end

%% Train knn 

knn_number = input('The number of K ?: '); %knn deðeri seçmelisin 
distance=0; 
counter=size(trainingVector,1); %Number of Total Train Data 
neighbors=zeros(1,knn_number); 
tagged=zeros(h,w); %Classified Image  
 for x=1:h %Calculating Euclidean Distance 
  for y=1:w           
      for z=1:counter             
          for band=1:spec_size         
              distance = distance + (hyperdata(x,y,band)- hyperdata(trainingData_row(z,1),trainingData_col(z,1),band))^2;  
          end
          dist(z,1)=sqrt(distance);  
          distance=0;    
      end
      [v , index]=sort(dist(:,1));     
      for k=1:knn_number      
          neighbors(k)=trainingVectorLabel(index(k),1);    
      end
  tagged(x,y)=mode(neighbors); % Selecting the most frequent class tag               
  end
 end
 
 figure;  
 subplot(1,2,1); imagesc(tagged); title('KNN Classified Hyperspectral Image'); 
 subplot(1,2,2); imagesc(hyperdata_gt); title('Groundtruth');   
 
 %% for knn
 
  %CALCULATING CLASSIFICATION SUCCESS RATE 
 true_positive=0;
 false_positive=0;  

 for x=1:h       
     for y=1:w    
         if(hyperdata_gt(x,y)~=0)            
             if(tagged(x,y) == hyperdata_gt(x,y))       
                 true_positive = true_positive+1; 
              else
                 false_positive = false_positive+1;
             end
         end
     end
 end
 
success_rate= true_positive*100 / (true_positive + false_positive); 
print=['Success Rate of KNN Classification = %',num2str(success_rate)]; 
disp(print)
