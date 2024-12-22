% 道路イメージを白黒に変更し，道路のみを抽出するプログラム
clear; close all;
%% 道路情報を畳み込むためのエクセルデータの読み込み
data = readtable("test2.xlsx");

% データを数値型に変換
data = table2array(data);

%% 道路画像データをインプット
img_in = double(imread("road_image2.jpg"));

figure()
subplot(1,2,1);
imshow(uint8(img_in));

% カラー画像を白黒に変更
img_gray = 0.299*img_in(:,:,1) + 0.587*img_in(:,:,2) + 0.114*img_in(:,:,3);

subplot(1,2,2);
imshow(uint8(img_gray));

%% 道路画像から道路のみを抽出
img_road = img_gray .* data;

figure()
subplot(1,2,1);
imshow(uint8(img_gray));
subplot(1,2,2);
imshow(uint8(img_road));
