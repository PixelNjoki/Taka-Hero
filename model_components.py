import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set device (for potentially loading to CPU if trained on GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformations (must be defined in model_components.py)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class WasteImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a blank image if loading fails
            return torch.zeros((3, 224, 224)), self.labels[idx]

class WasteImageClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(WasteImageClassifier, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class TextUrgencyClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(TextUrgencyClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in list(self.bert.parameters())[:-12]:
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(pooled_output)


class MultiModalPriorityClassifier(nn.Module):
    def __init__(self, num_priority_classes=4):
        super(MultiModalPriorityClassifier, self).__init__()

        # Image feature extractor
        self.image_model = models.efficientnet_b0(pretrained=True)
        num_image_features = self.image_model.classifier[1].in_features
        self.image_model.classifier = nn.Identity()

        # Text feature extractor
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Freeze most layers
        for param in list(self.image_model.parameters())[:-10]:
            param.requires_grad = False
        for param in list(self.text_model.parameters())[:-6]:
            param.requires_grad = False

        # Fusion layer
        combined_features = num_image_features + 768  # EfficientNet + DistilBERT

        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_priority_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # Extract image features
        image_features = self.image_model(image)

        # Extract text features
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Concatenate features
        combined = torch.cat([image_features, text_features], dim=1)

        # Classify priority
        priority_logits = self.fusion(combined)

        return priority_logits

class SolutionGenerator:
    def __init__(self):
        self.solutions = {
            'Critical': {
                'medical': [
                    "Immediately cordon off the area to prevent public access",
                    "Contact specialized medical waste disposal team",
                    "Alert local health authorities",
                    "Deploy hazmat-trained personnel with proper PPE",
                    "Arrange for biomedical waste incinerator disposal"
                ],
                'hazardous': [
                    "Evacuate immediate area and establish safety perimeter",
                    "Contact environmental protection agency",
                    "Deploy hazardous materials response team",
                    "Arrange containment and specialized disposal",
                    "Monitor environmental impact"
                ],
                'chemical': [
                    "Secure area and prevent access",
                    "Contact chemical emergency response team",
                    "Identify chemical type for proper handling",
                    "Arrange specialized hazmat disposal",
                    "Test surrounding soil and water"
                ]
            },
            'Urgent': {
                'blocking': [
                    "Deploy cleanup crew within 24 hours",
                    "Arrange heavy equipment for large waste removal",
                    "Clear drainage systems to prevent flooding",
                    "Set up temporary barriers to prevent further dumping",
                    "Investigate illegal dumping source"
                ],
                'accumulating': [
                    "Schedule immediate waste collection",
                    "Deploy multiple collection vehicles",
                    "Increase collection frequency for this area",
                    "Add temporary collection points"
                ],
                'electronic': [
                    "Arrange e-waste recycling specialist",
                    "Safely extract valuable materials",
                    "Dispose batteries and hazardous components properly",
                    "Partner with certified e-waste recyclers"
                ]
            },
            'Important': {
                'plastic': [
                    "Schedule waste collection within 3 days",
                    "Sort and send to recycling facility",
                    "Deploy standard cleanup crew",
                    "Install additional bins in area"
                ],
                'general': [
                    "Add to regular collection schedule",
                    "Send municipal cleanup team",
                    "Increase bin capacity in area",
                    "Post waste disposal guidelines"
                ]
            },
            'Secondary': {
                'minor': [
                    "Include in next scheduled collection round",
                    "Send maintenance crew when available",
                    "Monitor for accumulation",
                    "Community volunteer cleanup possible"
                ],
                'organic': [
                    "Collect during regular rounds",
                    "Consider composting program",
                    "Educate vendors on waste management"
                ]
            }
        }

    def generate_solution(self, priority, waste_type, description):
        """Generate contextual solution suggestions"""
        solutions = []

        # Get priority-level solutions
        priority_solutions = self.solutions.get(priority, {})

        # Try to match specific waste type
        if waste_type in priority_solutions:
            solutions.extend(priority_solutions[waste_type])
        else:
            # Use general solutions for this priority
            for key, sols in priority_solutions.items():
                solutions.extend(sols[:2])  # Take first 2 from each category

        # Add context-specific suggestions based on description
        description_lower = description.lower()
        if 'water' in description_lower or 'river' in description_lower:
            solutions.insert(0, "Prevent water contamination - prioritize waterway clearing")
        if 'school' in description_lower or 'children' in description_lower:
            solutions.insert(0, "Child safety priority - expedite removal near educational facilities")
        if 'market' in description_lower:
            solutions.append("Coordinate with market authorities for ongoing management")

        return list(dict.fromkeys(solutions[:5]))  # Return unique top 5 suggestions


class WastePriorityPredictor:
    def __init__(self, model_path='models/best_priority_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Ensure all necessary classes are accessible when loading
        self.model = MultiModalPriorityClassifier(num_priority_classes=4)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}. Using rule-based classification.")

        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.transform = val_transform
        self.priority_classes = ['Critical', 'Urgent', 'Important', 'Secondary']
        self.priority_colors = {'Critical': 'red', 'Urgent': 'orange', 'Important': 'black', 'Secondary': 'blue'}
        self.solution_generator = SolutionGenerator()

        # Rule-based fallback keywords
        self.critical_keywords = ['medical', 'toxic', 'hazardous', 'chemical', 'dangerous',
                                  'poisonous', 'leaking', 'hospital', 'needles', 'syringes']
        self.urgent_keywords = ['blocking', 'large', 'tons', 'truck', 'illegal dump',
                               'drain', 'waterway', 'river', 'flooding']
        self.important_keywords = ['street', 'road', 'public', 'park', 'market', 'scattered']

    def rule_based_priority(self, description, image_path=None):
        """Fallback rule-based priority classification"""
        desc_lower = description.lower()

        # Check for critical keywords
        if any(keyword in desc_lower for keyword in self.critical_keywords):
            return 'Critical', 0.95

        # Check for urgent keywords
        if any(keyword in desc_lower for keyword in self.urgent_keywords):
            return 'Urgent', 0.85

        # Check for important keywords
        if any(keyword in desc_lower for keyword in self.important_keywords):
            return 'Important', 0.75

        # Default to secondary
        return 'Secondary', 0.65

    def predict(self, image_path, description):
        """Predict priority for a waste report"""
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Tokenize description
            encoding = self.tokenizer(
                description,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Model prediction
            with torch.no_grad():
                outputs = self.model(image_tensor, input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                predicted_priority = self.priority_classes[predicted_idx.item()]
                confidence_score = confidence.item()

            # If confidence is low, use rule-based approach
            # Note: This threshold might need tuning after actual training
            if confidence_score < 0.6:
                print(f"Low model confidence ({confidence_score:.2f}). Falling back to rule-based.")
                predicted_priority, confidence_score = self.rule_based_priority(description, image_path)

        except Exception as e:
            print(f"Model prediction error: {e}. Using rule-based classification.")
            predicted_priority, confidence_score = self.rule_based_priority(description, image_path)

        # Detect waste type from description (simple keyword matching)
        waste_type = self.detect_waste_type(description)

        # Generate solutions
        solutions = self.solution_generator.generate_solution(
            predicted_priority, waste_type, description
        )

        result = {
            'priority': predicted_priority,
            'confidence': float(confidence_score),
            'color': self.priority_colors[predicted_priority],
            'waste_type': waste_type,
            'solutions': solutions,
            'status': 'Pending'
        }

        return result

    def detect_waste_type(self, description):
        """Detect waste type from description using keywords"""
        desc_lower = description.lower()

        waste_keywords = {
            'medical': ['medical', 'hospital', 'syringe', 'needle', 'bandage'],
            'hazardous': ['hazardous', 'toxic', 'chemical', 'dangerous', 'poisonous'],
            'electronic': ['electronic', 'e-waste', 'computer', 'phone', 'battery'],
            'plastic': ['plastic', 'bottle', 'bag', 'container'],
            'organic': ['organic', 'food', 'fruit', 'vegetable'],
            'metal': ['metal', 'can', 'aluminum', 'steel'],
            'glass': ['glass', 'bottle'],
            'paper': ['paper', 'cardboard'],
        }

        for waste_type, keywords in waste_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                return waste_type

        return 'general'

