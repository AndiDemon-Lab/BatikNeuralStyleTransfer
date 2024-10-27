from torchvision import transforms

class Transform:
    def __init__(self):
        self.transforms = {
            'vgg19': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'inception_v3': transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'resnet50': transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

    def get_transform(self, model_name):
        if model_name in self.transforms:
            return self.transforms[model_name]
        else:
            raise ValueError("Model name must be 'vgg19', 'inception_v3', or 'resnet50'.")
