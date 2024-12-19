%% 縦ベクトル画像データをブロックに戻し，そのブロック結合して画像データにするプログラム

%% エクセル上のテーブルを読み取る
t = readtable('a3_output.xlsx');

% データを数値型に変換（例: double）
data = table2array(t);

%% 縦ベクトルをブロック画像として保存
for i = 1:256
    % 16×16行列に変換
    block(:,:,i) = reshape(data(:,i), [16, 16])';
    filename = sprintf("block_out%d.bmp", i);
    filepath = fullfile("/Blocks_out", filename);
    % 白黒画像として保存
    imwrite(block(:,:,i), filepath);
end

%% ブロックデータを画像データとして保存
% ブロック数とサイズ
block_size = 16;
num_blocks = 256;
blocks_per_row = sqrt(num_blocks);
blocks_per_col = sqrt(num_blocks);

% 元の画像サイズ
image_height = block_size * blocks_per_col;
image_width = block_size * blocks_per_row;

% 元の画像を初期化
original_image = zeros(image_height, image_width, 'uint8');

% ブロックを読み込んで配置
count = 1;
for i = 1:blocks_per_col
    for j = 1:blocks_per_row
        % ブロックファイル名
        filename = sprintf("block_out%d.bmp", count);
        filepath = fullfile("Blocks_out", filename);
        
        % ブロックを読み込む
        block = imread(filepath);
        % 元の画像に配置
        original_image((i-1)*block_size+1:i*block_size, (j-1)*block_size+1:j*block_size) = block;
        count = count + 1;
    end
end

% 復元した画像を保存
imwrite(original_image, 'output.bmp');

% 復元した画像を表示
figure()
imshow(original_image, []);
title('Reconstructed Image');
