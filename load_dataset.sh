#!/bin/bash

# 正确的bash数组创建语法（注意：等号两边不能有空格）
git_dataset[0]="https://github.com/Jwoo5/ecg-qa.git"
zip_dataset[0]="https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
# 如果有更多数据集，可以继续添加
# git_dataset[1]="https://github.com/example/dataset2.git"
# git_dataset[2]="https://github.com/example/dataset3.git"

# 或者使用更简洁的方式创建数组
# git_dataset=("https://github.com/Jwoo5/ecg-qa.git" "其他数据集URL...")

cd dataset

# 遍历数组并下载每个git仓库
echo "开始下载Git数据集..."
for i in "${!git_dataset[@]}"; do
    url="${git_dataset[$i]}"
    echo "正在下载Git数据集 $((i+1)): $url"
    
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

echo "Git数据集下载完成！"

# 遍历数组并下载每个ZIP文件
echo "开始下载ZIP数据集..."
for i in "${!zip_dataset[@]}"; do
    url="${zip_dataset[$i]}"
    echo "正在下载ZIP数据集 $((i+1)): $url"
    
    # 提取文件名
    filename=$(basename "$url")
    # 如果URL没有文件扩展名，使用默认名称
    if [[ "$filename" != *.zip ]]; then
        filename="dataset_$((i+1)).zip"
    fi
    
    # 提取目录名（去掉.zip扩展名）
    dir_name=$(basename "$filename" .zip)
    
    # 检查目录是否已存在
    if [ -d "$dir_name" ]; then
        echo "目录 $dir_name 已存在，跳过下载"
    else
        # 下载ZIP文件
        echo "正在下载 $filename ..."
        wget -O "$filename" "$url"
        
        if [ $? -eq 0 ]; then
            echo "✓ 成功下载: $filename"
            
            # 检查文件是否确实是ZIP格式
            if file "$filename" | grep -q "Zip archive"; then
                echo "正在解压 $filename ..."
                
                # 创建目录并解压
                mkdir -p "$dir_name"
                unzip -q "$filename" -d "$dir_name"
                
                if [ $? -eq 0 ]; then
                    echo "✓ 成功解压到: $dir_name"
                    # 删除ZIP文件以节省空间
                    rm "$filename"
                    echo "✓ 已删除临时文件: $filename"
                else
                    echo "✗ 解压失败: $filename"
                fi
            else
                echo "✗ 下载的文件不是有效的ZIP格式: $filename"
                # 重命名为.html或.txt以便检查内容
                mv "$filename" "${filename%.zip}.html"
                echo "文件已重命名为 ${filename%.zip}.html 以便检查"
            fi
        else
            echo "✗ 下载失败: $url"
        fi
    fi
done

echo "所有数据集下载完成！"
