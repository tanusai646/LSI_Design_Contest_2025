clear; close all;

% 1. 画像の読み込みと前処理
img = imread('road_image2.jpg');
img_resized = imresize(img, [512, 512]);

figure;
subplot(2, 3, 1);
imshow(img_resized);
title('1. Original Image');

% 2. HSV色空間変換
hsvImg = rgb2hsv(img_resized);

% 道路部分に対応する色範囲のマスク作成（灰色～黒色を対象）
gray_mask = (hsvImg(:,:,2) < 0.3) & (hsvImg(:,:,3) > 0.2);

subplot(2, 3, 2);
imshow(gray_mask);
title('2. Gray Mask (HSV)');

% 3. グレースケール変換とエッジ検出
grayImg = rgb2gray(img_resized);
edges = edge(grayImg, 'Canny');

subplot(2, 3, 3);
imshow(edges);
title('3. Edge Detection (Canny)');

% 4. 形態学的処理によるノイズ除去と領域強調
se = strel('disk', 5); % 構造要素を定義
processed_mask = imdilate(gray_mask, se); % 膨張処理
processed_mask = imerode(processed_mask, se); % 収縮処理

subplot(2, 3, 4);
imshow(processed_mask);
title('4. Morphological Processing');

% 5. マスク適用で道路部分を抽出
road_only = bsxfun(@times, img_resized, cast(processed_mask, 'like', img_resized));

processed_mask_dou = double(processed_mask);
subplot(2, 3, 5);
imshow(road_only);
road_only_double = double(road_only);
title('5. Extracted Road Region');

filename='test2.xlsx';% ファイル名　拡張子がないと自動的に"txt"になる
A=magic(512);% 書き込むデータ
sheet_n='ABC';%書き込むシート
range_n='A1';%書き込む位置 省略した場合は左上がA1になる
writematrix(A,filename,'Sheet',sheet_n,'Range',range_n);%エクセルファイルに書き出し 

% 6. 全体結果の表示
subplot(2, 3, 6);
imshowpair(img_resized, road_only, 'montage');
title('6. Original vs Extracted');