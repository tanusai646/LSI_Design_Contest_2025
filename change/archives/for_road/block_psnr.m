% ブロック毎のPSNRを比較するプログラム
% 
clear;
%% フォルダ指定
%block_num = readtable("road_only_block.xlsx");
%block_num = table2array(block_num);

block_all = 1024;

block_PSNR = [];
test_num = [];

%% ブロック毎のPSNRの比較
for i = 1:block_all
    % 元の画像ブロック
    filename_1 = sprintf("fullblock_test%d.bmp", i);
    filepath_1 = fullfile("fullBlocks_test", filename_1); 
    % VAE通過後の画像ブロック
    filename_2 = sprintf("block_test_out%d.bmp", i);
    filepath_2 = fullfile("Blocks_test_out", filename_2);

    % ブロックを読み込む(本来はblock_numを用いる予定)
    if exist(filepath_2, 'file') == 2
        % ブロックを読み込む
        block_1 = double(im2gray(imread(filepath_1)));
        block_2 = double(im2gray(imread(filepath_2)));

        diff = abs((block_1 - block_2).^2);
        num = 0;
        % MSE, PSNRの測定
        for j = 1:16
            for k = 1:16
                num = num+diff(j,k);
            end
        end
        MSE = num / (16*16);
        PSNR = 10*log10((255)^2/MSE);
        
        block_PSNR = [block_PSNR, PSNR];
        test_num = [test_num, i];
    else
        % ファイルが存在しない場合の処理
        fprintf('File not found: %s\n', filepath_2); % ファイルがないことを通知
        block_PSNR = [block_PSNR, 50];
    end
end

block_PSNR_mat = reshape(block_PSNR, 32, 32);
block_PSNR_matt = block_PSNR_mat';
writematrix(block_PSNR_matt, "block_PSNR.xlsx");
test_num = reshape(test_num, 32, 32)';

