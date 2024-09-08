import os
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, Image as IPyImage
import ipywidgets as widgets

# Load image features và thông tin từ file .npy
image_features = np.load('result/keyframes_features.npy')
image_info = np.load('result/keyframes_info.npy', allow_pickle=True)

image_folder = "keyframes/"

# Tải mô hình CLIP và processor từ Hugging Face
model = CLIPModel.from_pretrained('clip_model/')
processor = CLIPProcessor.from_pretrained('clip_model/')

def text_to_features(text_list):
    inputs = processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.numpy()

# Hàm để hiển thị ảnh khi người dùng nhấp vào tên ảnh trong danh sách
def show_image(image_path):
    if os.path.exists(image_path):
        display(IPyImage(image_path))
    else:
        print(f"Image file '{image_path}' does not exist.")

# Yêu cầu người dùng nhập từ khóa, phân cách bởi dấu ";"
keywords_input = input("Nhập các từ khóa, phân cách bởi dấu ';': ").strip()

if not keywords_input:
    print("Bạn phải nhập ít nhất một từ khóa!")
else:
    keywords = [keyword.strip() for keyword in keywords_input.split(';')]

    # Chuyển đổi từ khóa thành đặc trưng văn bản
    text_features = text_to_features(keywords)

    # Tính toán độ tương đồng cosine giữa đặc trưng văn bản và đặc trưng hình ảnh
    similarities = cosine_similarity(text_features, image_features)

    # Lấy chỉ số của 50 ảnh phù hợp nhất
    top_indices = np.argsort(similarities[0])[::-1][:50]

    # Tạo danh sách lựa chọn cho hình ảnh
    image_selection = widgets.Select(
        options=[
            (f"Rank {i+1}: Folder '{image_info[index][0]}', Image '{image_info[index][1]}' - Similarity Score: {similarities[0][index]:.4f}", 
             os.path.join(image_folder, image_info[index][0], image_info[index][1] + ".jpg")) 
            for i, index in enumerate(top_indices)
        ],
        description='Chọn ảnh:',
        layout={'width': 'max-content'},
        style={'description_width': 'initial'}
    )

    # Kết nối lựa chọn hình ảnh với hàm hiển thị ảnh
    def on_image_select(change):
        show_image(change['new'])

    image_selection.observe(on_image_select, names='value')

    # Hiển thị danh sách lựa chọn
    display(image_selection)
