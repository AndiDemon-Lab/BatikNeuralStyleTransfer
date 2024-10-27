import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

class ImageHandler:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mean = torch.FloatTensor([[[0.485, 0.456, 0.406]]])
        self.std = torch.FloatTensor([[[0.229, 0.224, 0.225]]])

    def load_image(self, img_path, transform):
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        return image

    def gram_matrix(self, X):
        n, c, h, w = X.shape
        X = X.view(n*c, h*w) # Flattening
        G = torch.mm(X, X.t())
        G = G.div(n*c*h*w) # Normalization
        return G

    def draw_styled_image(self, output):
        styled_image = output[0].permute(1, 2, 0).cpu().detach()
        styled_image = styled_image * self.std + self.mean

        styled_image.clamp_(0, 1)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(styled_image)
        plt.axis("off")
        plt.pause(0.01)
        return styled_image
    
    def imshow(self, tensor, title=None):
        image = tensor.cpu().clone()  # clone the tensor to not change the original
        image = image.squeeze(0)      # remove the batch dimension
        image = transforms.ToPILImage()(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause to display the image

    def save_image(self, image, path):
        image = image.detach().cpu() 
        image = (image * 255).clamp(0, 255).byte() 
        if image.dim() == 4: 
            image = image.squeeze(0) 
        
        pil_image = Image.fromarray(image.permute(1, 2, 0).numpy())
        pil_image.save(path)

    def calculate_ssim(self, img1, img2, win_size=11):
        img1 = img1.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
        img2 = img2.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
        
        img1 = img1 * self.std.cpu().numpy() + self.mean.cpu().numpy()
        img2 = img2 * self.std.cpu().numpy() + self.mean.cpu().numpy()
        
        img1 = (img1 * 255).astype('uint8')
        img2 = (img2 * 255).astype('uint8')
        
        ssim_value, _ = ssim(img1, img2, win_size=win_size, channel_axis=2, full=True)
        return ssim_value