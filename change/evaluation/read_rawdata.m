clear;
% RAWファイルを読み込み
fileID = fopen('block_test401.raw', 'rb');
data = fread(fileID, 'uint8');
fclose(fileID);

data_1616 = reshape(data, 16, 16)';
% 4バイトごとに1バイトを除去
%data = reshape(data, 4, []);  % 4バイトの行列に変換
%data = data(1:3, :);         % 最初の3バイト（RGB）を抽出
%data = data(:);              % 1次元に戻す

% 新しいRAWファイルとして保存
%fileID = fopen('output_rgb.raw', 'wb');
%fwrite(fileID, data, 'uint8');
%fclose(fileID);