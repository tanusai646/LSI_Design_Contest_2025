% 縦ベクトル画像データをブロックに戻す
% そのブロック結合して画像データにする
% 元の画像と出力画像の比較を行う
% 出力はPSNRとBSE

clear;
%% エクセル上のテーブルを読み取る
t = readtable('fulla3_output.xlsx');
% データを数値型に変換（例: double）
data = table2array(t);

block_num = readtable("road_only_block.xlsx");
block_num = table2array(block_num);
block_num_size = size(block_num,2);

%% 縦ベクトルをブロック画像として保存
for i = 1:1024
    % 16×16行列に変換
    %num = block_num(1,i);
    block(:,:,i) = reshape(data(:,i), [16, 16])';
    filename = sprintf("block_test_out%d.bmp", i);
    filepath = fullfile("/Blocks_test_out", filename);
    % 白黒画像として保存
    imwrite(block(:,:,i), filepath);
end

%% ブロックデータを画像データとして保存
% ブロック数とサイズ
block_size = 16;
num_blocks = 1024;
blocks_per_row = sqrt(num_blocks);
blocks_per_col = sqrt(num_blocks);

% 元の画像サイズ
image_height = block_size * blocks_per_col;
image_width = block_size * blocks_per_row;

% 元の画像を初期化
recon_image = zeros(image_height, image_width, 'double');

% ブロックを読み込んで配置
count = 1;
for i = 1:blocks_per_col
    for j = 1:blocks_per_row
        % ブロックファイル名
        filename = sprintf("block_test_out%d.bmp", count);
        filepath = fullfile("Blocks_test_out", filename);
           

        % ブロックを読み込む(本来はblock_numを用いる予定)
        if exist(filepath, 'file') == 2
            % ブロックを読み込む
            block = double(im2gray(imread(filepath)));
        else
            % ファイルが存在しない場合の処理
            block = zeros(block_size, block_size, 'double'); % 空のブロックを代入
            %fprintf('File not found: %s\n', filepath); % ファイルがないことを通知
        end
        % 元の画像に配置
        recon_image((i-1)*block_size+1:i*block_size, (j-1)*block_size+1:j*block_size) = block;
        count = count + 1;
    end
end


% 復元した画像を保存
imwrite(uint8(recon_image), 'output_test.bmp');
original_image = double(imread("recon_image.bmp"));

difference = abs(original_image-recon_image);

%% 評価データ作成
MSE = sum(sum(abs(original_image(:,:) - recon_image(:,:)).^2))/(image_width*image_height);
PSNR = 10*log10((255)^2/MSE);
fprintf("MSE = %.3f , PSNR = %.3f [dB]\n", MSE, PSNR);

%% 復元した画像を表示
figure()
subplot(1,3,1);
imshow(uint8(original_image));
title('Original Image');
subplot(1,3,2);
imshow(uint8(recon_image));
title('Reconstructed Image');
subplot(1,3,3);
imshow(uint8(abs(original_image-recon_image)));