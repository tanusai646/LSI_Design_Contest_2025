clc, clear, close all;


% 画像データの生成
img = [1 1 1; 0 0 0; 1 1 1];

% figureの作成
figure;

% フレームの生成と表示
for i = 1:100
    % 画像データの変更
    tempImg = img;
    tempImg(1, 1) = i/100;

    % 画像を表示
    imshow(tempImg, 'InitialMagnification','fit');

    % 描画の更新
    drawnow;

    % 遅延 (0.1秒)
    pause(0.1);
end