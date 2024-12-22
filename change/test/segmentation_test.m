% 画像サイズとカーネル初期化
X = rand(32, 32); % 入力画像（例: ランダムな32x32画像）
t = double(X > 0.5); % 教師ラベル（閾値で分割）
w = rand(3, 3) * 0.01; % 初期重み（3x3カーネル）
b = 0; % 初期バイアス
eta = 0.01; % 学習率
epoch = 10; % エポック数

% モデル学習
[w, b, output, E] = SimpleSegmentation(X, t, w, b, eta, epoch);

% 結果の表示
figure;
subplot(1, 3, 1); imshow(X, []); title('入力画像');
subplot(1, 3, 2); imshow(t, []); title('教師ラベル');
subplot(1, 3, 3); imshow(output, []); title('セグメンテーション結果');
