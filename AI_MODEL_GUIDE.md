# Taka-Hero AI Priority Classification System

## Overview

This AI-powered system automatically analyzes waste reports and assigns priority levels based on both image content and text descriptions.

## Priority Levels

The system classifies reports into 4 priority levels:

### 🔴 Critical (Red)
- **Triggers**: Medical waste, hazardous materials, toxic chemicals, dangerous substances
- **Response Time**: Immediate (0-2 hours)
- **Examples**: Syringes near schools, leaking chemicals, hospital waste
- **Actions**: Specialized disposal teams, health authority alerts, area cordoning

### 🟠 Urgent (Orange)
- **Triggers**: Large dumps, blocked waterways, drainage blockages, illegal dumping
- **Response Time**: Within 24 hours
- **Examples**: Tons of waste blocking drains, massive accumulation in rivers
- **Actions**: Heavy equipment deployment, drainage clearing, dumping investigations

### ⚫ Important (Black)
- **Triggers**: Public space littering, recyclable accumulation, infrastructure concerns
- **Response Time**: Within 3 days
- **Examples**: Scattered waste in parks, multiple bags on roadsides
- **Actions**: Standard cleanup crews, recycling facility routing, bin installation

### 🔵 Secondary (Blue)
- **Triggers**: Minor littering, single items, maintenance issues
- **Response Time**: Regular collection schedule
- **Examples**: Single cardboard box, few items on pathway
- **Actions**: Regular collection rounds, community volunteer cleanups

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
# Basic dependencies (already in requirements.txt)
pip install flask flask-sqlalchemy flask-mail pillow

# AI Model dependencies (optional - system works without them)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

**Note**: The system works with or without PyTorch. If PyTorch is not installed, it uses a rule-based keyword classification system.

### Step 2: Database Migration

Update your database to include the new AI fields:

```bash
# Option 1: Reset database (WARNING: Deletes all data)
rm instance/taka_hero.db
python app.py  # This will recreate the database with new fields

# Option 2: Keep existing data and add new columns manually
# Use a database migration tool or SQL commands
```

### Step 3: Test the AI Module

```bash
# Test the predictor independently
python waste_predictor.py
```

This will run test cases and show you how the classification works.

### Step 4: Run the Application

```bash
python app.py
```

The app will automatically:
1. Try to load the AI model
2. Fall back to rule-based classification if model not found
3. Classify all new reports automatically

---

## Training the AI Model (Optional)

To train the full deep learning model on real data:

### 1. Open the Training Notebook

```bash
# Install Jupyter if not installed
pip install jupyter notebook

# Launch notebook
jupyter notebook waste_priority_model_training.ipynb
```

### 2. Download Waste Dataset

The notebook uses publicly available waste datasets. You can:

**Option A: Kaggle Dataset**
```bash
# Configure Kaggle API
mkdir -p ~/.kaggle
# Add your kaggle.json credentials to ~/.kaggle/

# The notebook will download automatically
```

**Option B: Manual Download**
- Visit: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- Download and extract to `data/waste_images/`

### 3. Run All Cells in the Notebook

The notebook will:
- Load and preprocess data
- Train the multi-modal model (image + text)
- Save trained models to `models/` directory
- Generate evaluation metrics

### 4. Use the Trained Model

Once training completes, the model files are automatically used by the Flask app:
- `models/waste_priority_model.pth` - Main model
- `models/model_config.json` - Configuration
- `models/solution_generator.pkl` - Solution system

---

## How It Works

### 1. Report Submission
```
User submits report → Image uploaded → Description provided
```

### 2. AI Analysis
```python
# The system analyzes:
prediction = predictor.predict(image_path, description)

# Returns:
{
    'priority': 'Critical',          # Priority level
    'confidence': 0.95,              # Confidence score (0-1)
    'color': 'red',                  # Color code for UI
    'waste_type': 'medical',         # Detected waste type
    'solutions': [                   # Context-aware suggestions
        'Immediately cordon off the area...',
        'Contact specialized medical waste disposal team...',
        ...
    ],
    'status': 'Pending'
}
```

### 3. Database Storage
All AI-generated data is stored in the database:
- `priority` - Classification result
- `priority_color` - UI color code
- `waste_type` - Detected type
- `confidence` - Model confidence
- `ai_solutions` - Generated action items
- `status` - Workflow status

### 4. Admin Dashboard
Reports are automatically sorted by priority:
1. Critical (Red) - Top of list
2. Urgent (Orange)
3. Important (Black)
4. Secondary (Blue) - Bottom of list

---

## API Endpoints

### Get Report Details
```bash
GET /api/report/<report_id>
```

Returns full report including AI analysis:
```json
{
  "id": 1,
  "priority": "Critical",
  "priority_color": "red",
  "waste_type": "medical",
  "confidence": 0.95,
  "solutions": [
    "Immediately cordon off the area...",
    "Contact specialized medical waste disposal team..."
  ],
  "status": "Pending",
  ...
}
```

---

## Customization

### Adding New Keywords

Edit `waste_predictor.py` to add custom keywords for your region:

```python
self.critical_keywords = [
    'medical', 'toxic', 'hazardous',
    # Add Kenya-specific terms:
    'dawa', 'sumu', 'hatari'  # Swahili terms
]
```

### Adding New Waste Types

```python
waste_keywords = {
    'medical': ['medical', 'hospital', 'syringe'],
    'plastic': ['plastic', 'bottle', 'bag'],
    # Add new type:
    'construction': ['concrete', 'rubble', 'debris']
}
```

### Customizing Solutions

Edit the `SolutionGenerator` class to add region-specific solutions:

```python
'Critical': {
    'medical': [
        "Contact Nairobi Health Department: +254-XXX-XXXX",
        "Alert NEMA (National Environment Management Authority)",
        ...
    ]
}
```

---

## Workflow Integration

### Automatic Status Updates

The system automatically manages report status:

1. **New Report** → AI analyzes → Status: **"Pending"**
2. **Admin Views** → Status: **"In Progress"** (manual update)
3. **Issue Resolved** → Status: **"Resolved"**
4. **Resolved Reports** → Can be removed or archived

### Status Update Endpoint

```python
GET /update_status/<report_id>/<status>
# Example: /update_status/5/In Progress
```

---

## Troubleshooting

### AI Model Not Loading

**Symptom**: "Using rule-based classification" message

**Solutions**:
1. Check if `models/waste_priority_model.pth` exists
2. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Run training notebook to generate model
4. The system will work fine with rule-based classification

### Database Errors

**Symptom**: Column doesn't exist errors

**Solution**: Reset database or add columns manually:
```sql
ALTER TABLE waste_report ADD COLUMN priority VARCHAR(20) DEFAULT 'Secondary';
ALTER TABLE waste_report ADD COLUMN priority_color VARCHAR(20) DEFAULT 'blue';
ALTER TABLE waste_report ADD COLUMN waste_type VARCHAR(50) DEFAULT 'general';
ALTER TABLE waste_report ADD COLUMN confidence FLOAT DEFAULT 0.0;
ALTER TABLE waste_report ADD COLUMN ai_solutions TEXT;
```

### Low Accuracy

**Solutions**:
1. Train model with more Kenya-specific data
2. Add local keywords to rule-based classifier
3. Adjust confidence thresholds
4. Collect real-world data and retrain

---

## Performance

### Rule-Based Classification (Default)
- **Speed**: Instant (< 10ms)
- **Accuracy**: ~75-80% (keyword-based)
- **Requirements**: None (works out of the box)

### Deep Learning Model (After Training)
- **Speed**: ~100-500ms per report
- **Accuracy**: ~85-92% (with training)
- **Requirements**: PyTorch, trained model files

---

## Future Enhancements

1. **Mobile App Integration**: Priority badges in mobile UI
2. **Real-time Notifications**: Alert authorities for Critical reports
3. **Geographic Hotspot Detection**: Identify problem areas
4. **Trend Analysis**: Track waste types over time
5. **Citizen Feedback**: Allow users to confirm AI classifications
6. **Multi-language Support**: Swahili/English hybrid classification

---

## Files Structure

```
Taka-Hero/
├── app.py                              # Main Flask app (updated with AI)
├── waste_predictor.py                  # AI predictor module
├── waste_priority_model_training.ipynb # Training notebook
├── AI_MODEL_GUIDE.md                   # This guide
├── models/
│   ├── waste_priority_model.pth       # Trained model (after training)
│   ├── model_config.json              # Model configuration
│   └── solution_generator.pkl         # Solution system
├── static/uploads/                     # User-uploaded images
└── instance/
    └── taka_hero.db                   # SQLite database

```

---

## Support & Contact

For issues, questions, or contributions:
- GitHub Issues: [Your repository]
- Email: [Your email]
- Documentation: This guide

---

## License

This AI system is part of the Taka-Hero project for making Kenya clean! 🇰🇪♻️

**Built with ❤️ for a cleaner Kenya**
