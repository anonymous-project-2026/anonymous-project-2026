
import os

def generate_config():
    # Use relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "res", "omm_filter_tpl.txt")
    output_path = os.path.join(script_dir, "feature_config.py")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return

    libs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            libs.append(line)
            
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Auto-generated configuration file\n\n")
        f.write("FILTER_LIBRARIES = [\n")
        for lib in libs:
            f.write(f"    '{lib}',\n")
        f.write("]\n\n")
        
        f.write("FILTER_FUNCTIONS = [\n")
        f.write("    '<init>',\n")
        f.write("    '<clinit>'\n")
        f.write("]\n")
        
    print(f"Successfully generated {output_path} with {len(libs)} libraries.")

if __name__ == "__main__":
    generate_config()
