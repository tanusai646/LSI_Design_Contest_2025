%% 画像の読み取り
clc, clear; close all;

% 指定された画像を開く
% 画像をモノクロに変更
image = double(rgb2gray(imread("input.bmp")))/255.0; % 0~255を0~1にする
SizeX = size(image,2) / 16;
SizeY = size(image,1) / 16;


% 画像を8×8のブロックに変える
count = 1;
for i = 1:SizeY
    for j = 1:SizeX
        % 16x16のブロックを切り出す
        block = image((i-1)*16+1:i*16, (j-1)*16+1:j*16); 
        % ファイル名を作成
        filename = sprintf("block%d.bmp", count);
        filepath = fullfile("/Blocks", filename);
        
        % ブロックを保存
        imwrite(block, filepath);
        count = count+1;
    end
end

for k = 1:3
    filename = sprintf("block%d.bmp", k);
    filepath = fullfile("Blocks", filename);
    figure(k);
    imshow(filepath, 'InitialMagnification','fit');
end
 