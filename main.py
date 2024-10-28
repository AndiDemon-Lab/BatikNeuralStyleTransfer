import os
import json
import torch
import time
from src.models import NeuralStyleTransfer
from src.nst_utils import ImageHandler
from src.criterion import Criterion
from src.data_validation import TrainRequest
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model option ['vgg19', 'resnet50', inception_v3(still error)]
model_name = "resnet50"
pretrained_weights_path = None
nst = NeuralStyleTransfer(model_name, pretrained_weights_path=pretrained_weights_path).to(device)
criterion = Criterion()
image_handler = ImageHandler()

def load_existing_metadata(metadata_filename):
    if os.path.exists(metadata_filename):
        with open(metadata_filename, 'r') as json_file:
            metadata = json.load(json_file)
            if "sessions" not in metadata:
                metadata["sessions"] = [] 
            return metadata
    return {"sessions": []} 

def train(request: TrainRequest):
    # load content and style images
    content_image = image_handler.load_image(request.content_image_path, image_handler.transform).to(device)
    style_image = image_handler.load_image(request.style_image_path, image_handler.transform).to(device)

    # prepare output image
    output = content_image.clone()
    output.requires_grad = True
    optimizer = optim.AdamW([output], lr=0.05)
    
    # extract features
    content_features = nst(content_image, layers=["4", "8"])
    style_features = nst(style_image, layers=["4", "8"])

    max_epochs = 2500
    print(f'--------------------- Start Training ---------------------')
    generated_image_name = ""
    
    # create output directory specific to the model
    model_output_dir = f"outputs/generate_{model_name}"
    os.makedirs(model_output_dir, exist_ok=True)

    # load existing metadata
    metadata_filename = "outputs/metadata.json"
    metadata = load_existing_metadata(metadata_filename)

    # prepare new session metadata
    new_session = {
        "model_name": model_name,
        "is_finetuned": pretrained_weights_path is not None,
        "content_image": request.content_image_path,
        "style_image": request.style_image_path,
        "generated_images": [],
        "loss_values": [],
        "ssim_values": []
    }

    for epoch in range(1, max_epochs + 1):
        output_features = nst(output, layers=["4", "8"])
        loss = criterion.criterion(content_features, style_features, output_features, output_features, style_weight=1e6)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch:5} | Loss: {loss.item():.5f}")
            ssim_value = image_handler.calculate_ssim(output, content_image)
            print(f"Epoch: {epoch:5} | SSIM: {ssim_value:.5f}\n")
        
        # save output images at specific epochs
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if epoch in {800, 1600, 2500}:
            output_image_path = f"{model_output_dir}/output_epoch_{epoch}_{timestamp}.png"
            image_handler.save_image(output, output_image_path)
            generated_image_name = f"output_epoch_{epoch}.png"

            # add generated image metadata
            new_session["generated_images"].append({
                "epoch": epoch,
                "filename": output_image_path,
                "timestamp": timestamp
            })
            # Store loss and SSIM for this epoch
            new_session["loss_values"].append(loss.item())
            new_session["ssim_values"].append(ssim_value)

    # append new session to existing metadata
    metadata["sessions"].append(new_session)

    # save updated metadata back to the JSON file
    with open(metadata_filename, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    return {"message": "Training completed!", "generated_image_name": generated_image_name}

def upload_and_train(content_image_path: str, style_image_path: str):
    # Create TrainRequest object and initiate training
    train_request = TrainRequest(
        content_image_path=content_image_path,
        style_image_path=style_image_path
    )
    result = train(train_request)
    print(result)

def main(content_image_path: str, style_image_path: str):
    upload_and_train(content_image_path, style_image_path)

if __name__ == "__main__":
    content_image_path = "data/train_dikit/batik-barong/1.jpg" 
    style_image_path = "data/train_dikit/batik-kawung/1.jpg" 
    
    main(content_image_path, style_image_path)