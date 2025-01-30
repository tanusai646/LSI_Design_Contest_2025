%% 今まで作成したプログラムを結合させたものを作成

%% 初期データリセット
clear; close all;

%% 道路画像を入力し，道路情報のみを保存する部分
% 道路画像(GRBカラー)
img_in = double(imread("road_image4.bmp"));


% カラー画像を白黒に変換
img_gray = 0.299*img_in(:,:,1) + 0.587*img_in(:,:,2) + 0.114*img_in(:,:,3);

imwrite(uint8(img_gray), "road_image4_gray.bmp");

figure()
% 元画像の表示
subplot(1,3,1);
imshow(uint8(img_in));
title("original image");
% 白黒画像の表示
subplot(1,3,2);
imshow(uint8(img_gray));
title("gray image");


%% 画像をブロック化
% 画像全てと道路のみの画像をブロック化する
% ブロックに変えるサイズ設定(変更可能にする予定)
size_b = 16;

size_x = size(img_gray,2)/size_b;
size_y = size(img_gray,1)/size_b;

% 初期値設定
count = 1;
block_all = zeros(size_b, size_b, size_x*size_y);

% グレー画像をブロック化
for i = 1:size_y
    for j = 1:size_x
        block_all(:,:,count) = img_gray((i-1)*size_b+1:i*size_b, (j-1)*size_b+1:j*size_b);
        count = count + 1;
        flag = 0;
    end
end

%% VAEの部分
% VAEの学習は別のファイルで行っている
% この部分は学習済みVAEを利用している

rng(2025);          % Set random seed.

block_all_size = size(block_all,3);

% すべての画像をNew64_2に格納
New64_2(:,:,:) = block_all/255.0;

%% テストデータの入力
for i = 1:1024
    TestData(:,i) = reshape(New64_2(:,:,i)', size_b*size_b,1);
    TLabelData(:,i) = TestData(:,i);
end

% VAEの情報ロード
load("vae_model.mat");


X = TestData;
[z2_mean,z2_var, a2_mean, a2_var,z,z3,a3] = Neuralnetwork2_forward_VAE(X,w12_mean,w12_var,w23,b2_mean,b2_var,b3);

for i = 1:block_all_size
    % 16×16行列に変換
    %num = block_num(1,i);
    block_all_ret(:,:,i) = reshape(a3(:,i), [size_b, size_b])';
    % 正規化
    block_all_ret(:,:,i) = block_all_ret(:,:,i) * 255;
end

% ブロックを結合
count = 1;
for i = 1:size_y
    for j = 1:size_x
        % 元の画像に配置
        img_ret((i-1)*size_b+1:i*size_b, (j-1)*size_b+1:j*size_b) = block_all_ret(:,:,count);
        count = count + 1;
    end
end

subplot(1,3,3);
imshow(uint8(img_ret));
title('Reconstructed Image');

imwrite(uint8(img_ret),"output_test.bmp");

MSE = sum(sum(abs(img_ret(:,:) - img_gray(:,:)).^2))/(size_b*size_b);
PSNR = 10*log10((255)^2/MSE);
fprintf("MSE = %.3f , PSNR = %.3f [dB]\n", MSE, PSNR);

block_PSNR = [];

%% ブロック毎のPSNRの比較
for i = 1:block_all_size
    block_1 = block_all(:,:,i);
    block_2 = block_all_ret(:,:,i);

    diff = abs((block_1 - block_2).^2);
    PSNR_c = 0;
    % MSE, PSNRの測定
    for j = 1:size_b
        for k = 1:size_b
            PSNR_c = PSNR_c+diff(j,k);
        end
    end
    MSE = PSNR_c / (size_b*size_b);
    PSNR = 10*log10((255)^2/MSE);
    
    block_PSNR = [block_PSNR, PSNR];
end

block_PSNR_mat = reshape(block_PSNR, size_y, size_x)';
writematrix(block_PSNR_mat, "block_PSNR.xlsx");

figure();
colormap(jet);
imagesc(block_PSNR_mat);
colorbar;
clim([0, 40]);

figure();
histogram(block_PSNR);

fprintf("program complete\n");