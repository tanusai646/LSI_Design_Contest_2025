% 道路画像をブロックにして保存するプログラム
% 道路の部分のみをブロックにし，その他の部分はブロックとして保存しない
%% 画像の読み取り
clc, clear; close all;

% 指定された画像を開く
% road_only2.bmpはグレー画像
image = double(imread("road_only2.bmp"))/255.0; % 0~255を0~1にする
figure()
subplot(1,2,1);
imshow(image,'InitialMagnification','fit');
%imwrite(image, 'input_gray.bmp')

SizeX = size(image,2) / 16;
SizeY = size(image,1) / 16;

block_num = []; % 引き渡すブロックの番号を読み取り

%% 画像を16×16のブロックに変える
count = 1;
flag = 0;
for i = 1:SizeY
    for j = 1:SizeX
        % 16x16のブロックを切り出す
        block = image((i-1)*16+1:i*16, (j-1)*16+1:j*16); 
        % ファイル名を作成
        filename = sprintf("block_test%d.bmp", count);
        filepath = fullfile("/Blocks_test", filename);
        % ブロックの左上，左下，右上または右下にブラックがないか確認
        if block(1,1) == 0
            flag = 1;
        elseif block(1,16) == 0
            flag = 1;
        elseif block(16,1) == 0
            flag = 1;
        elseif block(16,16) == 0
            flag = 1;
        end
        % ブロックを保存
        if flag == 0
            imwrite(block, filepath);
            block_num = [block_num, count];
        end
        count = count+1;
        flag = 0;
    end
end

%% ブロック番号をエクセルに仮保存
writematrix(block_num, "road_only_block.xlsx")

%for k = 1:3
%   filename = sprintf("block%d.bmp", k);
%   figure(k);
%   imshow(filepath, 'InitialMagnification','fit');
%end

%% ブロックを再配置テスト
% ブロック数とサイズ
block_size = 16;    %ブロックのサイズ
num_blocks = 1024;  %ブロックの数
blocks_per_row = sqrt(num_blocks);
blocks_per_col = sqrt(num_blocks);

% 元の画像サイズ
image_height = block_size * blocks_per_col;
image_width = block_size * blocks_per_row;

% 元の画像を初期化
recon_image = zeros(image_height, image_width, 'double');

count = 1;
for i = 1:blocks_per_col
    for j = 1:blocks_per_row
        % ブロックファイル名
        filename = sprintf("block_test%d.bmp", count);
        filepath = fullfile("Blocks_test", filename);
           

        % ブロックを読み込む(本来はblock_numを用いる予定)
        if exist(filepath, 'file') == 2
            % ブロックを読み込む
            block = double(im2gray(imread(filepath)));
        else
            % ファイルが存在しない場合の処理
            block = zeros(block_size, block_size, 'double'); % 空のブロックを代入
            fprintf('File not found: %s\n', filepath); % ファイルがないことを通知
        end
        % 元の画像に配置
        recon_image((i-1)*block_size+1:i*block_size, (j-1)*block_size+1:j*block_size) = block;
        count = count + 1;
    end
end
writematrix(recon_image, "recon_block.xlsx")
imwrite(uint8(recon_image),"recon_image.bmp");
subplot(1,2,2);
imshow(uint8(recon_image));