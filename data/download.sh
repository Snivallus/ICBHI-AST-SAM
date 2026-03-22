URL="https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"
DATA_DIR="./data"
ZIP_FILE="$DATA_DIR/ICBHI_final_database.zip"
SRC_DIR="$DATA_DIR/ICBHI_final_database"

# 创建目录
mkdir -p "$DATA_DIR"

# 下载数据集
curl -L -k "$URL" -o "$ZIP_FILE"

# 解压数据集
unzip -q "$ZIP_FILE" -d "$DATA_DIR"

# 删除压缩包
rm "$ZIP_FILE"

# 下载训练集 / 测试集分割文件
URL="https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt"
TEXT_FILE="$DATA_DIR/ICBHI_Challenge_train_test.txt"
curl -L -k "$URL" -o "$TEXT_FILE"