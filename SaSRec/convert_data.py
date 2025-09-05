def convert_to_long_format(input_file, output_file):
    """
    将序列数据文件从宽格式转换为长格式。

    Args:
        input_file (str): 输入文件的路径 (宽格式)。
        output_file (str): 输出文件的路径 (长格式)。
    """
    print(f"开始转换文件: {input_file}")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            # 移除行首和行尾的空白字符，并按空格分割
            parts = line.strip().split()
            
            # 忽略空行或不合规的行
            if len(parts) < 2:
                print(f"警告: 第 {line_num} 行数据不足，已跳过。")
                continue

            # 第一个元素是用户ID
            user_id = parts[0]
            
            # 后续所有元素都是物品ID
            item_ids = parts[1:]

            # 为每个物品ID创建新的一行
            for item_id in item_ids:
                f_out.write(f"{user_id} {item_id}\n")
    
    print(f"文件转换成功！已将结果保存至: {output_file}")

if __name__ == '__main__':
    # 定义输入和输出文件名
    input_filename = 'sequential_data.txt'
    output_filename = 'data_long_format.txt'

    try:
        # 执行转换
        convert_to_long_format(input_filename, output_filename)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_filename}' 未找到。请确保它与脚本在同一目录下。")
    except Exception as e:
        print(f"发生未知错误: {e}")