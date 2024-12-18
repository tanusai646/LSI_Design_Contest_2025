% すべてのFigureハンドルを取得
figHandles = findall(0, 'Type', 'figure');

% 保存するPDFファイル名を指定
pdfFileName = 'all_figures.pdf';

% Figureが存在するか確認
if ~isempty(figHandles)
    % 1つ目のFigureをPDFとして保存
    exportgraphics(figHandles(1), pdfFileName, 'Append', false); 

    % 2つ目以降のFigureをPDFに追加
    for i = 2:length(figHandles)
        exportgraphics(figHandles(i), pdfFileName, 'Append', true);
    end

    fprintf('すべてのFigureが %s に保存されました。\n', pdfFileName);
else
    disp('保存するFigureが存在しません。');
end