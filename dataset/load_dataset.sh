#!/bin/bash

# 正确的bash数组创建语法（注意：等号两边不能有空格）
git_dataset[0]="https://github.com/Jwoo5/ecg-qa.git"
# 如果有更多数据集，可以继续添加
# git_dataset[1]="https://github.com/example/dataset2.git"
# git_dataset[2]="https://github.com/example/dataset3.git"

# 或者使用更简洁的方式创建数组
# git_dataset=("https://github.com/Jwoo5/ecg-qa.git" "其他数据集URL...")

cd dataset

# 遍历数组并下载每个git仓库
echo "开始下载数据集..."
for i in "${!git_dataset[@]}"; do
    url="${git_dataset[$i]}"
    echo "正在下载数据集 $((i+1)): $url"
    
    # 提取仓库名称作为目录名
    repo_name=$(basename "$url" .git)
    
    # 检查目录是否已存在
    if [ -d "$repo_name" ]; then
        echo "目录 $repo_name 已存在，跳过下载"
    else
        # 克隆仓库
        git clone "$url"
        if [ $? -eq 0 ]; then
            echo "✓ 成功下载: $repo_name"
        else
            echo "✗ 下载失败: $url"
        fi
    fi
done

echo "数据集下载完成！"
