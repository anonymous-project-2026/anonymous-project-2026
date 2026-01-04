import os
import logging
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def find_icon_file(decompiled_apk_dir, resource_type, resource_name):
    """
    根据资源类型和名称查找图标文件（可能在 res/mipmap* 或 res/drawable* 等目录）。
    返回匹配的文件路径列表。
    """
    icon_files = []
    extensions = [".png", ".jpg", ".webp"]

    res_dir = os.path.join(decompiled_apk_dir, "res")
    if os.path.exists(res_dir):
        for root, dirs, files in os.walk(res_dir):
            for dir_name in dirs:
                if dir_name.startswith(resource_type):
                    icon_dir = os.path.join(root, dir_name)
                    try:
                        for file in os.listdir(icon_dir):
                            if file.startswith(resource_name) and any(file.endswith(ext) for ext in extensions):
                                icon_files.append(os.path.join(icon_dir, file))
                    except FileNotFoundError:
                        # 某些情况下目录可能在遍历期间被移除，忽略即可
                        continue
    return icon_files

def select_best_icon(icon_files):
    """
    从图标文件列表中选择最合适的（按分辨率优先）。
    """
    resolution_order = ["xxxhdpi", "xxhdpi", "xhdpi", "hdpi", "mdpi"]
    for res in resolution_order:
        for f in icon_files:
            if res in f:
                return f
    return icon_files[0] if icon_files else None

def collect_apk_icons(decompiled_apk_dir):
    """
    从 AndroidManifest.xml 中读取 application 的 android:icon 属性，
    然后查找对应资源文件并返回优选的图标文件路径（或 None）。
    """
    manifest_path = os.path.join(decompiled_apk_dir, "AndroidManifest.xml")
    if not os.path.exists(manifest_path):
        logging.warning(f"AndroidManifest.xml 未找到: {manifest_path}")
        return None

    try:
        tree = ET.parse(manifest_path)
        root = tree.getroot()
        namespace = "{http://schemas.android.com/apk/res/android}"

        application = root.find("application")
        if application is not None:
            icon_resource = application.get(f"{namespace}icon")
            if icon_resource and icon_resource.startswith("@"):
                # icon_resource 例如 "@mipmap/ic_launcher" 或 "@drawable/ic_launcher"
                try:
                    resource_type, resource_name = icon_resource[1:].split("/")
                except ValueError:
                    logging.warning(f"无法解析 icon 资源: {icon_resource} in {manifest_path}")
                    return None
                icon_files = find_icon_file(decompiled_apk_dir, resource_type, resource_name)
                if icon_files:
                    return select_best_icon(icon_files)
    except ET.ParseError as e:
        logging.error(f"解析 AndroidManifest.xml 失败: {manifest_path}, 错误: {e}")
    except Exception as e:
        logging.exception(f"collect_apk_icons 未知错误: {e}")

    return None

if __name__ == "__main__":
    base_directory = "/newdisk/liuzhuowu/analysis/data/decom"  # 修改为你的根目录
    log_file = os.path.join(base_directory, "collect_icons.log")
    setup_logging(log_file)

    # 收集所有 APK 反编译目录（假设结构为 base/<group>/<apk_dir>/）
    all_apk_paths = []
    if os.path.isdir(base_directory):
        for group_folder_name in os.listdir(base_directory):
            group_folder_path = os.path.join(base_directory, group_folder_name)
            if os.path.isdir(group_folder_path):
                for apk_folder_name in os.listdir(group_folder_path):
                    apk_folder_path = os.path.join(group_folder_path, apk_folder_name)
                    if os.path.isdir(apk_folder_path):
                        all_apk_paths.append(apk_folder_path)

    all_apk_paths = sorted(all_apk_paths)

    # 遍历每个 apk 目录，查找图标并复制到该目录下的 icon 子文件夹
    for apk_path in tqdm(all_apk_paths, desc="Processing APKs", unit="apk"):
        try:
            icon_file = collect_apk_icons(apk_path)
            if icon_file:
                icon_dir = os.path.join(apk_path, "icon")
                os.makedirs(icon_dir, exist_ok=True)

                dest_path = os.path.join(icon_dir, os.path.basename(icon_file))
                # 若已存在同名文件则先移除（可按需改为跳过或重命名）
                if os.path.exists(dest_path):
                    os.remove(dest_path)

                shutil.copy2(icon_file, dest_path)
                logging.info(f"Copied icon for {apk_path} -> {dest_path}")
            else:
                logging.info(f"No icon found for {apk_path}")
        except Exception as e:
            logging.exception(f"处理 {apk_path} 时发生错误: {e}")
