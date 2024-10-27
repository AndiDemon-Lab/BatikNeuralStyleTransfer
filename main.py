import os
import torch
from src.models import NeuralStyleTransfer
from src.nst_utils import ImageHandler
from src.criterion import Criterion
from src.data_validation import TrainRequest
from PIL import Image
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = NeuralStyleTransfer().to(device)
criterion = Criterion()
image_handler = ImageHandler()

def train(request: TrainRequest):
    content_image = image_handler.load_image(request.content_image_path, image_handler.transform).to(device)
    style_image = image_handler.load_image(request.style_image_path, image_handler.transform).to(device)

    output = content_image.clone()
    output.requires_grad = True
    optimizer = optim.AdamW([output], lr=0.05)
    
    content_features = model(content_image, layers=["4", "8"])
    style_features = model(style_image, layers=["4", "8"])

    max_epochs = 2500
    print(f'---------------------start training---------------------')
    generated_image_name = ""
    for epoch in range(1, max_epochs + 1):
        output_features = model(output, layers=["4", "8"])
        loss = criterion.criterion(content_features, style_features, output_features, output_features, style_weight=1e6)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch:5} | Loss: {loss.item():.5f}")
            ssim_value = image_handler.calculate_ssim(output, content_image)
            print(f"Epoch: {epoch:5} | SSIM: {ssim_value:.5f}\n")
        
        if epoch in {800, 1600, 2500}:
            output_image_path = f"outputs/output_epoch_{epoch}.png"
            image_handler.save_image(output, output_image_path)
            generated_image_name = f"output_epoch_{epoch}.png"

    return {"message": "Training completed!", "generated_image_name": generated_image_name}

def upload_and_train(content_image_path: str, style_image_path: str):
    train_request = TrainRequest(
        content_image_path=content_image_path,
        style_image_path=style_image_path
    )
    result = train(train_request)
    print(result)

if __name__ == "__main__":
    content_image_path = "data/raw_batik_v2/raw_batik_v2/test/Aceh_Pintu_Aceh/10005.jpg" 
    style_image_path = "data/raw_batik_v2/raw_batik_v2/test/Bali_Barong/200022.jpg" 
    
    os.makedirs("uploads", exist_ok=True)
    
    upload_and_train(content_image_path, style_image_path)