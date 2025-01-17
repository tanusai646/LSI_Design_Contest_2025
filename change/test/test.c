#include <stdio.h>
#include <stdlib.h>

#define WIDTH 16
#define HEIGHT 16

int main() {
    FILE *file;
    unsigned char image_data[HEIGHT*WIDTH]; // 16×16のグレースケール画像
    const char *filename = "block_test401.raw";

    // ファイルを開く
    file = fopen(filename, "rb");
    if (file == NULL) {
        perror("ファイルを開けません");
        return 1;
    }

    // データを読み込む
    size_t read_count = fread(image_data, sizeof(unsigned char), WIDTH * HEIGHT, file);
    if (read_count != WIDTH * HEIGHT) {
        fprintf(stderr, "画像データの読み込みに失敗しました\n");
        fclose(file);
        return 1;
    }

    // ファイルを閉じる
    fclose(file);

    // 読み込んだデータを確認する（例として画素値を出力）
    for (int i = 0; i < HEIGHT*WIDTH; i++) {
        printf("%3d ", image_data[i]); // 各画素値を出力
        if((i+1) % 16 == 0){
            printf("\n");
        }
    }

    double image_data_i[256];
    for(int i = 0; i < 256; i++){
        image_data_i[i] = (double)image_data[i] / 255.0;
        printf("%5.3f ", image_data_i[i]);
        if((i+1) % 16 == 0){
            printf("\n");
        }
    }

    return 0;
}
