"""
Waste Priority Prediction Module for Taka-Hero

This module provides AI-powered priority classification for waste reports.
It analyzes both images and text descriptions to assign priority levels:
- Critical (Red): Hazardous/medical waste requiring immediate action
- Urgent (Orange): Large dumps, blockages, environmental threats
- Important (Black): Public space waste, recyclables
- Secondary (Blue): Minor littering, single items

The module also generates context-aware solution suggestions.
"""

import os
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    from transformers import DistilBertTokenizer, DistilBertModel
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("Warning: PyTorch not installed. Using rule-based classification only.")


class SolutionGenerator:
    """Generates context-aware solution suggestions based on waste type and priority"""
    
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


if DEEP_LEARNING_AVAILABLE:
    class MultiModalPriorityClassifier(nn.Module):
        """Deep learning model for multi-modal priority classification"""
        
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


class WastePriorityPredictor:
    """
    Main predictor class for waste report priority classification
    
    This class analyzes waste images and text descriptions to:
    1. Classify priority level (Critical, Urgent, Important, Secondary)
    2. Identify waste type
    3. Generate solution suggestions
    4. Assign color codes for visualization
    """
    
    def __init__(self, model_path='models/waste_priority_model.pth'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained PyTorch model (optional)
        """
        self.priority_classes = ['Critical', 'Urgent', 'Important', 'Secondary']
        self.priority_colors = {
            'Critical': 'red',
            'Urgent': 'orange', 
            'Important': 'black',
            'Secondary': 'blue'
        }
        self.solution_generator = SolutionGenerator()
        
        # Rule-based fallback keywords
        self.critical_keywords = [
            'medical', 'toxic', 'hazardous', 'chemical', 'dangerous',
            'poisonous', 'leaking', 'hospital', 'needles', 'syringes',
            'burning', 'fire', 'explosive', 'radioactive'
        ]
        self.urgent_keywords = [
            'blocking', 'large', 'tons', 'truck', 'illegal dump',
            'drain', 'waterway', 'river', 'flooding', 'massive',
            'accumulating', 'growing', 'expanding'
        ]
        self.important_keywords = [
            'street', 'road', 'public', 'park', 'market', 'scattered',
            'multiple', 'bags', 'containers'
        ]
        
        # Initialize deep learning model if available
        if DEEP_LEARNING_AVAILABLE:
            try:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = MultiModalPriorityClassifier(num_priority_classes=4)
                
                if os.path.exists(model_path):
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"Model loaded from {model_path}")
                    self.model_loaded = True
                else:
                    print(f"Model file not found at {model_path}. Using rule-based classification.")
                    self.model_loaded = False
                
                self.model.to(self.device)
                self.model.eval()
                
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            except Exception as e:
                print(f"Error initializing deep learning model: {e}")
                print("Falling back to rule-based classification.")
                self.model_loaded = False
        else:
            self.model_loaded = False
            print("Using rule-based classification (PyTorch not available)")
    
    def rule_based_priority(self, description, image_path=None):
        """
        Fallback rule-based priority classification using keyword matching
        
        Args:
            description: Text description of the waste report
            image_path: Path to image (not used in rule-based approach)
            
        Returns:
            tuple: (priority_level, confidence_score)
        """
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
    
    def detect_waste_type(self, description):
        """
        Detect waste type from description using keyword matching
        
        Args:
            description: Text description of the waste report
            
        Returns:
            str: Detected waste type
        """
        desc_lower = description.lower()
        
        waste_keywords = {
            'medical': ['medical', 'hospital', 'syringe', 'needle', 'bandage', 'pharmaceutical'],
            'hazardous': ['hazardous', 'toxic', 'chemical', 'dangerous', 'poisonous', 'corrosive'],
            'electronic': ['electronic', 'e-waste', 'computer', 'phone', 'battery', 'appliance'],
            'plastic': ['plastic', 'bottle', 'bag', 'container', 'packaging'],
            'organic': ['organic', 'food', 'fruit', 'vegetable', 'compost'],
            'metal': ['metal', 'can', 'aluminum', 'steel', 'iron'],
            'glass': ['glass', 'bottle', 'jar'],
            'paper': ['paper', 'cardboard', 'newspaper'],
        }
        
        for waste_type, keywords in waste_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                return waste_type
        
        return 'general'
    
    def predict(self, image_path, description):
        """
        Predict priority for a waste report using image and text
        
        Args:
            image_path: Path to the waste image
            description: Text description of the waste
            
        Returns:
            dict: Prediction results including priority, confidence, solutions, etc.
        """
        # Try deep learning model first if available
        if DEEP_LEARNING_AVAILABLE and self.model_loaded:
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
                if confidence_score < 0.6:
                    predicted_priority, confidence_score = self.rule_based_priority(description, image_path)
                
            except Exception as e:
                print(f"Model prediction error: {e}. Using rule-based classification.")
                predicted_priority, confidence_score = self.rule_based_priority(description, image_path)
        else:
            # Use rule-based classification
            predicted_priority, confidence_score = self.rule_based_priority(description, image_path)
        
        # Detect waste type from description
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
    
    def predict_batch(self, reports):
        """
        Predict priorities for multiple reports
        
        Args:
            reports: List of dicts with 'image_path' and 'description' keys
            
        Returns:
            list: List of prediction results
        """
        results = []
        for report in reports:
            result = self.predict(report['image_path'], report['description'])
            results.append(result)
        return results


# Convenience function for easy import
def create_predictor(model_path='models/waste_priority_model.pth'):
    """
    Create and return a WastePriorityPredictor instance
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        WastePriorityPredictor: Initialized predictor
    """
    return WastePriorityPredictor(model_path)


if __name__ == "__main__":
    # Test the predictor
    print("Testing Waste Priority Predictor\n")
    print("=" * 80)
    
    predictor = create_predictor()
    
    test_cases = [
        {
            'description': 'Medical waste including syringes found near school playground',
            'expected': 'Critical'
        },
        {
            'description': 'Large pile of plastic waste blocking drainage system',
            'expected': 'Urgent'
        },
        {
            'description': 'Scattered bottles and cans in public park',
            'expected': 'Important'
        },
        {
            'description': 'Single cardboard box on sidewalk',
            'expected': 'Secondary'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        priority, confidence = predictor.rule_based_priority(test['description'])
        waste_type = predictor.detect_waste_type(test['description'])
        solutions = predictor.solution_generator.generate_solution(
            priority, waste_type, test['description']
        )
        
        print(f"\nTest Case {i}:")
        print(f"Description: {test['description']}")
        print(f"Expected: {test['expected']}")
        print(f"Predicted: {priority} (Confidence: {confidence:.1%})")
        print(f"Color: {predictor.priority_colors[priority]}")
        print(f"Waste Type: {waste_type}")
        print(f"Solutions:")
        for j, sol in enumerate(solutions, 1):
            print(f"  {j}. {sol}")
        print("-" * 80)
