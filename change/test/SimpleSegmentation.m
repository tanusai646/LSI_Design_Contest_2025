function [w, b, output, E] = SimpleSegmentation(X, t, w, b, eta, epoch)
    % X: 入力画像 (H x W)
    % t: 教師ラベル画像 (H x W)
    % w: 畳み込みカーネル (k x k)
    % b: バイアス (スカラー)
    % eta: 学習率
    % epoch: 学習エポック数
    
    % 初期化
    [H, W] = size(X);
    k = size(w, 1); % カーネルサイズ
    output = zeros(H, W); % 出力画像
    E = zeros(1, epoch); % 誤差記録

    % パディングサイズ
    pad = floor(k / 2);

    % エポックごとの学習ループ
    for e = 1:epoch
        % === フォワードプロパゲーション ===
        % 畳み込み演算
        conv_output = zeros(H, W);
        for i = 1+pad:H-pad
            for j = 1+pad:W-pad
                region = X(i-pad:i+pad, j-pad:j+pad); % 対応する領域
                conv_output(i, j) = sum(region .* w, 'all') + b; % 畳み込み
            end
        end
        
        % 活性化関数（シグモイド関数を使用）
        output = 1 ./ (1 + exp(-conv_output));

        % === 誤差計算 ===
        Erec = -t .* log(output) - (1 - t) .* log(1 - output); % クロスエントロピー誤差
        E(e) = sum(Erec, 'all') / numel(X);

        % === バックプロパゲーション ===
        % 出力層の誤差
        delta = output - t;

        % 畳み込みカーネルとバイアスの勾配計算
        dEdw = zeros(k, k);
        dEdb = sum(delta, 'all');
        for i = 1+pad:H-pad
            for j = 1+pad:W-pad
                region = X(i-pad:i+pad, j-pad:j+pad);
                dEdw = dEdw + delta(i, j) .* region; % 勾配計算
            end
        end

        % パラメータの更新
        w = w - eta * dEdw;
        b = b - eta * dEdb;
    end
end