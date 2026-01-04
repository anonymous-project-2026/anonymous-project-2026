import os
import shutil

def extract_so_files(apk_folder_path):
    """
    从反编译的APK文件夹中提取SO文件
    :param apk_folder_path: 反编译后的APK文件夹路径
    :return: 成功返回True，失败返回False
    """
    try:
        # 定义优先级顺序
        arch_priority = ['arm64-v8a', 'armeabi-v7a', 'armeabi', 'x86', 'x86_64']
        
        # 构建lib文件夹路径
        lib_folder = os.path.join(apk_folder_path, "lib")
        
        # 检查lib文件夹是否存在
        if not os.path.exists(lib_folder):
            print(f"  lib文件夹不存在!")
            return False
        
        # 构建目标文件夹路径
        so_files_folder = os.path.join(apk_folder_path, "so_files")
        
        # 删除已存在的so_files文件夹
        if os.path.exists(so_files_folder):
            try:
                shutil.rmtree(so_files_folder)
                print(f"  已删除现有的so_files文件夹")
            except Exception as e:
                print(f"  删除so_files文件夹时出错: {e}")
                return False
        
        # 创建新的so_files文件夹
        os.makedirs(so_files_folder, exist_ok=True)
        
        # 按优先级顺序查找并复制SO文件
        selected_arch = None
        copied_files = []
        
        for arch in arch_priority:
            arch_folder = os.path.join(lib_folder, arch)
            if os.path.exists(arch_folder) and os.path.isdir(arch_folder):
                # 检查文件夹中是否有SO文件
                so_files = [f for f in os.listdir(arch_folder) if f.endswith('.so')]
                if so_files:
                    selected_arch = arch
                    print(f"  找到架构文件夹: {arch}")
                    
                    # 复制所有SO文件
                    for so_file in so_files:
                        src_path = os.path.join(arch_folder, so_file)
                        dst_path = os.path.join(so_files_folder, so_file)
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                            copied_files.append(so_file)
                        except Exception as e:
                            print(f"  复制文件 {so_file} 时出错: {e}")
                    
                    # 找到一个有效的架构后就停止
                    break
        
        if not copied_files:
            print(f"  在所有架构文件夹中均未找到SO文件")
            return False
        
        print(f"  成功从 {selected_arch} 架构复制了 {len(copied_files)} 个SO文件")
        
        # 预处理：将所有SO文件转为txt
        _batch_convert_so_to_txt(so_files_folder)
        
        return True
        
    except Exception as e:
        print(f"  提取SO文件时出错: {e}")
        return False

def _batch_convert_so_to_txt(so_folder):
    """
    将文件夹下的所有 .so 文件转换为 .txt (十六进制文本)
    """
    if not os.path.exists(so_folder):
        return
    
    files = [f for f in os.listdir(so_folder) if f.endswith('.so')]
    if not files:
        return
        
    print(f"  正在将 {len(files)} 个SO文件转换为txt...")
    for f in files:
        so_path = os.path.join(so_folder, f)
        txt_path = so_path + ".txt"
        
        # 如果txt已存在，跳过
        if os.path.exists(txt_path):
            continue
            
        try:
            with open(so_path, 'rb') as rf:
                content = rf.read()
            # 转为十六进制字符串 (如 "7f454c46...")
            hex_str = content.hex()
            with open(txt_path, 'w', encoding='utf-8') as wf:
                wf.write(hex_str)
        except Exception as e:
            print(f"    转换SO文件失败 {f}: {e}")


def batch_extract_so_files(decom_base_directory):
    """
    批量处理所有APK文件夹中的SO文件提取
    :param decom_base_directory: 反编译APK的基础目录
    """
    all_apk_paths = []
    
    # 收集所有APK文件夹路径
    for group_folder_name in os.listdir(decom_base_directory):
        group_folder_path = os.path.join(decom_base_directory, group_folder_name)
        if os.path.isdir(group_folder_path):
            for apk_folder_name in os.listdir(group_folder_path):
                apk_folder_path = os.path.join(group_folder_path, apk_folder_name)
                if os.path.isdir(apk_folder_path):
                    all_apk_paths.append(apk_folder_path)
    
    all_apk_paths = sorted(all_apk_paths)
    
    print(f"找到 {len(all_apk_paths)} 个APK文件夹")
    print("开始批量提取SO文件...\n")
    
    success_count = 0
    fail_count = 0
    
    # 批量处理每个APK文件夹
    for i, apk_folder_path in enumerate(all_apk_paths, 1):
        print(f"[{i}/{len(all_apk_paths)}] 处理: {os.path.basename(apk_folder_path)}")
        
        if extract_so_files(apk_folder_path):
            success_count += 1
        else:
            fail_count += 1
        
        print()  # 空行分隔每个APK的处理结果
    
    # 输出统计结果
    print("=" * 50)
    print(f"批量处理完成!")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(all_apk_paths)}")

# 使用示例
if __name__ == "__main__":
    decom_base_directory = r"/newdisk/liuzhuowu/analysis/data/decom"
    
    if not os.path.isdir(decom_base_directory):
        print("基础目录不存在!")
        exit(1)
    
    batch_extract_so_files(decom_base_directory)