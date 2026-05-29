import os
target_dir = r"C:\Users\bowli\Desktop\playaround programs\post-storm damage\data\train\undamaged"
print("Files found:", os.listdir(target_dir)[:5] if os.path.exists(target_dir) else "Folder doesn't exist!")