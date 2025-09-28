#!/bin/bash

# 清空RaBitQ-Library的result和logger目录下所有.log文件的脚本
# 作者：Assistant
# 日期：$(date)

# 设置基础路径
BASE_PATH="/home/wbh/cppwork/RaBitQ-Library"
RESULT_PATH="$BASE_PATH/result"
LOGGER_PATH="$BASE_PATH/logger"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}开始清空RaBitQ-Library的.log文件...${NC}"

# 计数器
total_files=0
cleaned_files=0

# 函数：清空指定目录下的.log文件
clean_log_files() {
    local dir_path="$1"
    local dir_name="$2"
    
    echo -e "\n${YELLOW}正在处理 $dir_name 目录...${NC}"
    
    if [ ! -d "$dir_path" ]; then
        echo -e "${RED}目录不存在: $dir_path${NC}"
        return
    fi
    
    # 遍历所有数据集目录
    for dataset_dir in "$dir_path"/*; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            echo -e "  处理数据集: ${GREEN}$dataset_name${NC}"
            
            # 查找并清空.log文件
            local log_files=("$dataset_dir"/*.log)
            if [ -e "${log_files[0]}" ]; then
                for log_file in "${log_files[@]}"; do
                    if [ -f "$log_file" ]; then
                        local file_name=$(basename "$log_file")
                        local file_size=$(stat -c%s "$log_file" 2>/dev/null || echo "0")
                        
                        # 清空文件
                        > "$log_file"
                        
                        echo -e "    清空: ${GREEN}$file_name${NC} (原大小: ${file_size}字节)"
                        ((total_files++))
                        ((cleaned_files++))
                    fi
                done
            else
                echo -e "    ${YELLOW}未找到.log文件${NC}"
            fi
        fi
    done
}

# 清空result目录下的.log文件
clean_log_files "$RESULT_PATH" "result"

# 清空logger目录下的.log文件
clean_log_files "$LOGGER_PATH" "logger"

echo -e "\n${GREEN}============ 清理完成 ============${NC}"
echo -e "${GREEN}总共处理文件数: $total_files${NC}"
echo -e "${GREEN}成功清空文件数: $cleaned_files${NC}"

echo -e "\n${GREEN}脚本执行完成！${NC}"