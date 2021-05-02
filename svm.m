%% Cross Validation'suz hali.
clear; clc; close all;

%% Params
% User Params
perc_rate = 0.1;    %Egitim verisinden se�ilecek verinin orani % default 0.05
verbose = true;
% kernel_type = 'RBF';
kernel_type = 'polynomial';
% kernel_type = 'linear';
%% Load Images
hyperdata = load('indian_pines.mat');
% hyperdata = hyperdata.salinas;
hyperdata = hyperdata.indian_pines;
% hyperdata = hyperdata.paviaU; %anlad���m kadar�yla b�yle yap�nca benim hiperspektral g�r�nt�m� bir de�i�kene(matris) at�yorum

hyperdata_gt = load('indian_pines_gt.mat');
% hyperdata_gt = hyperdata_gt.salinas_gt; 
hyperdata_gt = hyperdata_gt.indian_pines_gt;
% hyperdata_gt = hyperdata_gt.paviaU_gt;
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
    NumOfClassElements(i) = double(sum(sum((hyperdata_gt == i)))); %ka� 1 ka� 2 .... ka� 9 var
end

%% ** %Perc_Rate data will select for training vector ** 
for i = 1 : 1 : NumOfClass
    perc_5 = floor(NumOfClassElements(i) * perc_rate);
    [row,col] = find(hyperdata_gt == i);
    
    %Maskedeki ilgili etiket degeri y�zde 5'e ulasmadigi s�rece �rnek al
    %mask� g�ncelliyo hyperdata_gt verileri ile gibi d���nebiliriz
    while(sum(sum(Mask == i)) ~= perc_5) 
        x = floor((rand() * (NumOfClassElements(i) - 1)) + 1);
        Mask(row(x),col(x)) = hyperdata_gt(row(x),col(x));
    end
end

%% Class's Nums

ClassSayisi = 0;
for i = 1 : 1 : NumOfClass
   ClassSayisi = ClassSayisi + sum(sum(Mask == i)); 
    %perc de�erine g�re ald�n ya onun i�inde ka� tane 1 2...9 var
    %mask i�inde 1 den ka� tane var + 2 den ka� tane var......9 dan ka�
    %tane var 
end

%% Created Train and Label Vector %%etiket ve e�itimmm miii

[trainingData_row,trainingData_col,values] = find(Mask ~= 0);
%zaten mask i�inde 0 olmayan de�erlerin row,col ve etiketlerini tutuyo  
trainingVector = zeros(ClassSayisi,spec_size); %hepsi i�in samples lar�  tutucak
trainingVectorLabel = zeros(ClassSayisi,1); %s�n�flar� tutucak

% ****  Training Vector & Training Vector Label  ****

for i = 1 : ClassSayisi
    trainingVector(i,:)      = hyperdata(trainingData_row(i),trainingData_col(i),:);
    trainingVectorLabel(i,1) = hyperdata_gt(trainingData_row(i),trainingData_col(i));
end

%% Train
tic;
classes = unique(trainingVectorLabel);
num_classes = numel(classes);
svms = cell(num_classes,1);

for k=1:NumOfClass
    if verbose
        fprintf(['Training Classifier ', num2str(classes(k)) ' of ', num2str(num_classes), '\n']);
    end
    class_k_label = trainingVectorLabel == classes(k);
    svms{k} = fitcsvm(trainingVector, class_k_label, 'Standardize',...
        true,'KernelScale', 'auto', 'KernelFunction', kernel_type, ...
        'CacheSize', 'maximal', 'BoxConstraint', 10);
end

%% %**********************Classify the test data**********************
for k=1:NumOfClass
    if verbose
        fprintf(['Classifying with Classifier ', num2str(classes(k)),...
            ' of ', num2str(num_classes), '\n']);
    end
    [~, temp_score] = predict(svms{k}, hypervector);
    score(:, k) = temp_score(:, 2);                     %Her satirin ilgili sutununa sinifla ilgili score degerini diz.
end
[~, est_label] = max(score, [], 2);
prediction_svm = im2uint8(zeros(h*w, 1));

for k=1:num_classes
    prediction_svm(find(est_label==k),:) = k;
end
prediction_svm = reshape(prediction_svm, [h, w, 1]);

z = find(hyperdata_gt == 0);
prediction_svm(z) = 0;


ERR = sum(sum( (prediction_svm ~= hyperdata_gt) ));
NumOfElements = sum(sum(NumOfClassElements(:)));
NumOfTrueElements = NumOfElements - ERR;
RATE = (NumOfTrueElements / NumOfElements)*100;

%% Results
fprintf(['\n','***********************************************************','\n']);
fprintf(['\tBasari Orani : ', num2str(RATE),'\n']);
fprintf(['***********************************************************','\n']);

figure , imshow(label2rgb(prediction_svm, @jet, [.5 .5 .5])) , title('Sonuc');
figure , imshow(label2rgb(hyperdata_gt, @jet, [.5 .5 .5])) , title('Orijinal');