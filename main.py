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

# HYPERPARAMETERS TO SET
# model option ['vgg19', 'vgg16' 'resnet50', 'resnet101' inception_v3(still error)]
model_name = "vgg19"
pretrained_weights_path = None
content_weight = 2
style_weight = 1e8
content_layers = ["4", "8"]
style_layers = ["4", "8"]

# instance
nst = NeuralStyleTransfer(model_name, pretrained_weights_path=pretrained_weights_path).to(device)
criterion = Criterion(content_weight=content_weight, style_weight=style_weight)
image_handler = ImageHandler()

# get metadata for generated images
def load_existing_metadata(metadata_filename):
    if os.path.exists(metadata_filename):
        with open(metadata_filename, 'r') as json_file:
            metadata = json.load(json_file)
            if "sessions" not in metadata:
                metadata["sessions"] = [] 
            return metadata
    return {"sessions": []} 

# train nst
def train(request: TrainRequest):
    # load content and style images
    content_image = image_handler.load_image(request.content_image_path, image_handler.transform).to(device)
    style_image = image_handler.load_image(request.style_image_path, image_handler.transform).to(device)

    # prepare output image
    output = content_image.clone()
    output.requires_grad = True
    optimizer = optim.AdamW([output], lr=0.05)
    
    # extract features
    content_features = nst(content_image, layers=content_layers)
    style_features = nst(style_image, layers=style_layers)

    max_epochs = 1000
    print(f'--------------------- Start Training ---------------------')
    generated_image_name = ""
    
    # create output directory specific to the model
    model_output_dir = f"outputs/generate_{model_name}"
    os.makedirs(model_output_dir, exist_ok=True)

    # load existing metadata
    metadata_filename = "outputs/new_metadata.json"
    metadata = load_existing_metadata(metadata_filename)

    # prepare new session metadata
    new_session = {
        "model_name": model_name,
        "is_finetuned": pretrained_weights_path is not None,
        "content_image": request.content_image_path,
        "style_image": request.style_image_path,
        "content_weight": content_weight,
        "style_weight": style_weight,
        "content_layers": content_layers,
        "style_layers": style_layers,
        "generated_images": [],
        "loss_values": [],
        "ssim_values": []
    }

    # Record start time
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        output_features = nst(output, layers=["4", "8"])
        loss = criterion.criterion(content_features, style_features, output_features, output_features)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch:5} | Loss: {loss.item():.5f}")
            ssim_value = image_handler.calculate_ssim(output, content_image)
            print(f"Epoch: {epoch:5} | SSIM: {ssim_value:.5f}\n")
        
        # save output images at specific epochs
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if epoch in {100, 500, 1000}:
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

    # Calculate training time and add it to metadata
    end_time = time.time()
    training_time = end_time - start_time
    new_session["training_time"] = training_time  # in seconds

    # append new session to existing metadata
    metadata["sessions"].append(new_session)

    # save updated metadata back to the JSON file
    with open(metadata_filename, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    return {"message": "Training completed!", "generated_image_name": generated_image_name, "training_time": training_time}

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

# # single generate
# if __name__ == "__main__":
#     content_image_path = "data\_content\sa.jpg" 
#     style_image_path = "data/train_dikit/batik-kawung/1.jpg" 
    
#     main(content_image_path, style_image_path)

# batch generate
if __name__ == "__main__":
    content_folder = "data/_content"
    style_folder = "data/_style"

    content_image_paths = [os.path.join(content_folder, f) for f in os.listdir(content_folder) if f.endswith((".jpg", ".png"))]
    style_image_paths = [os.path.join(style_folder, f) for f in os.listdir(style_folder) if f.endswith((".jpg", ".png"))]

    os.makedirs("outputs", exist_ok=True)

    for content_image_path in content_image_paths:
        for style_image_path in style_image_paths:
            print(f"Processing Content: {content_image_path}, Style: {style_image_path}")
            try:
                main(content_image_path, style_image_path)
            except Exception as e:
                print(f"Error processing {content_image_path} with {style_image_path}: {e}")
