clear;
close all;

%% 入力画像を窓関数にかけて，0か1に近い値にする

%% 画像の読み取り 
% 0~255までだから，それを255で割ることで0-1に落とし込む．
for i = 1:2
    filename = sprintf("block%d.bmp", i);
    filepath = fullfile("../Blocks", filename);
    New64(:,:,i) = double(im2gray(imread(filepath)))/255.0;
end

figure(100); 
subplot(2,1,1);
imshow(New64(:,:,1));
subplot(2,1,2);
imshow(New64(:,:,2));

imagex = 16; 
imagey=16;

for i = 1:2
    TrainData(:,i) = reshape(New64(:,:,i)', imagex*imagey,1);
    LabelData(:,i) = TrainData(:,i); 
end

%% シグモイド関数とロジック関数の定義
k = 10;
sigmod = @(x) 1 ./ (1 + -exp(-k * (-x+0.5)));
logit = @(y) 1/k * log(y ./ (1-y)) + 0.5;

%% シグモイド関数の適用
for i = 1:2
    outputM(:,i) = sigmod(LabelData(:,i));
end

%% 適用後の画像表示
figure(101);
for j = 1:size(outputM,2)
    subplot(size(outputM,2),1,j);
    imshow(reshape(outputM(:,j), [16 16])'); %サイズ変更
end

%% 元の画像に戻す
for i = 1:2
    recimage(:,i) = logit(outputM(:,i));
end

%% 適用後の画像表示
figure(102);
for j = 1:size(recimage,2)
    subplot(size(recimage,2),1,j);
    imshow(reshape(recimage(:,j), [16 16])'); %サイズ変更
end

