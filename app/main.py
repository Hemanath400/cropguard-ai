from pathlib import Path
import io

import tensorflow as tf
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import urllib.request

import os
from google import genai
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_URL = (
    "https://huggingface.co/Hemnath14/cropguard-resnet50/"
    "resolve/main/resnet50_tomato_finetuned_best.keras"
)

MODEL_PATH = BASE_DIR / "models" / "resnet50_tomato_finetuned_best.keras"

if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading CropGuard model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

print("STEP 1: Starting TensorFlow/model initialization")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("STEP 2: Model loaded successfully")
except Exception as e:
    print("MODEL LOAD ERROR:", repr(e))
    raise

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("Loading CropGuard AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="CropGuard AI",
    description="Tomato leaf disease detection API",
    version="1.0.0"
)


# ------------------------------------------------------------
# Static files
# ------------------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ------------------------------------------------------------
# Jinja2 templates
# ------------------------------------------------------------

templates = Jinja2Templates(
    directory=TEMPLATES_DIR
)


# ------------------------------------------------------------
# Disease classes
# ------------------------------------------------------------

CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy"
]

# ------------------------------------------------------------
# Disease Management Knowledge Base
# ------------------------------------------------------------

DISEASE_ADVICE = {

    "Bacterial Spot": {
        "description": "A bacterial disease that causes small dark spots and lesions on tomato leaves and fruit.",
        "symptoms": [
            "Small dark brown or black spots on leaves",
            "Yellowing around some lesions",
            "Spots may enlarge under favorable conditions"
        ],
        "immediate_actions": [
            "Remove severely infected leaves",
            "Avoid working with plants when foliage is wet",
            "Improve air circulation around plants"
        ],
        "prevention": [
            "Avoid overhead irrigation",
            "Keep foliage as dry as possible",
            "Use clean seeds and disease-free planting material",
            "Sanitize gardening tools"
        ],
        "treatment": [
            "Use locally approved copper-based or other registered bacterial disease treatments according to the product label",
            "Follow local agricultural extension recommendations"
        ]
    },

    "Early Blight": {
        "description": "A fungal disease that commonly produces dark lesions with concentric rings on older leaves.",
        "symptoms": [
            "Dark brown circular lesions",
            "Concentric ring patterns resembling a target",
            "Yellowing of surrounding leaf tissue",
            "Usually begins on older lower leaves"
        ],
        "immediate_actions": [
            "Remove heavily affected leaves",
            "Remove fallen infected plant debris",
            "Improve air circulation"
        ],
        "prevention": [
            "Avoid overhead watering",
            "Water at the base of plants",
            "Maintain adequate spacing",
            "Rotate crops where practical"
        ],
        "treatment": [
            "Use a locally registered fungicide when appropriate",
            "Follow the product label and local agricultural guidance"
        ]
    },

    "Late Blight": {
        "description": "A destructive disease that can spread rapidly under cool and humid conditions.",
        "symptoms": [
            "Large irregular dark lesions",
            "Rapid browning or collapse of leaves",
            "Dark lesions may occur on stems and fruit",
            "White fungal growth may appear under humid conditions"
        ],
        "immediate_actions": [
            "Remove severely infected plant material",
            "Separate affected plants when practical",
            "Avoid handling wet foliage"
        ],
        "prevention": [
            "Improve ventilation",
            "Avoid prolonged leaf wetness",
            "Monitor plants frequently during cool, wet weather"
        ],
        "treatment": [
            "Use an appropriate locally registered fungicide promptly when disease is confirmed",
            "Follow local agricultural extension recommendations"
        ]
    },

    "Leaf Mold": {
        "description": "A fungal disease favored by high humidity and poor air circulation.",
        "symptoms": [
            "Yellow patches on upper leaf surfaces",
            "Olive-green to brown growth on leaf undersides",
            "Older leaves are commonly affected first"
        ],
        "immediate_actions": [
            "Remove severely affected leaves",
            "Increase ventilation",
            "Reduce humidity around foliage"
        ],
        "prevention": [
            "Avoid overhead irrigation",
            "Provide adequate plant spacing",
            "Improve greenhouse ventilation when applicable"
        ],
        "treatment": [
            "Use a locally registered fungicide if necessary",
            "Follow product-label instructions"
        ]
    },

    "Septoria Leaf Spot": {
        "description": "A fungal leaf-spot disease that produces numerous small lesions, often on lower leaves.",
        "symptoms": [
            "Small circular spots",
            "Gray or tan centers",
            "Dark margins around lesions",
            "Tiny dark structures may appear within spots"
        ],
        "immediate_actions": [
            "Remove infected lower leaves",
            "Clean up fallen plant debris",
            "Improve air circulation"
        ],
        "prevention": [
            "Avoid overhead watering",
            "Keep foliage dry",
            "Use adequate plant spacing",
            "Practice crop rotation where possible"
        ],
        "treatment": [
            "Use a locally approved fungicide when appropriate",
            "Follow local agricultural recommendations"
        ]
    },

    "Spider Mites": {
        "description": "Tiny pests that feed on plant tissue and can cause stippling, yellowing and leaf decline.",
        "symptoms": [
            "Fine yellow or pale speckling",
            "Bronzing or drying of leaves",
            "Fine webbing may be visible",
            "Severe infestations can cause leaf drop"
        ],
        "immediate_actions": [
            "Inspect leaf undersides carefully",
            "Remove severely damaged leaves",
            "Isolate heavily infested plants when practical"
        ],
        "prevention": [
            "Monitor plants regularly",
            "Reduce excessive plant stress",
            "Encourage beneficial predatory insects where appropriate"
        ],
        "treatment": [
            "Use an appropriate registered miticide or other approved control when necessary",
            "Follow label directions carefully",
            "Avoid unnecessary broad-spectrum pesticide use"
        ]
    },

    "Target Spot": {
        "description": "A fungal disease that produces brown lesions with concentric rings and may cause leaf drop.",
        "symptoms": [
            "Brown circular lesions",
            "Concentric rings within lesions",
            "Yellowing around affected areas",
            "Leaf drop in severe cases"
        ],
        "immediate_actions": [
            "Remove severely affected leaves",
            "Remove infected plant debris",
            "Improve air circulation"
        ],
        "prevention": [
            "Avoid overhead irrigation",
            "Keep foliage dry",
            "Maintain adequate plant spacing",
            "Remove plant debris after harvest"
        ],
        "treatment": [
            "Use a locally registered fungicide when appropriate",
            "Follow product-label instructions"
        ]
    },

    "Tomato Yellow Leaf Curl Virus": {
        "description": "A viral disease commonly transmitted by whiteflies and associated with severe leaf curling and stunted growth.",
        "symptoms": [
            "Upward curling of leaves",
            "Yellowing of young leaves",
            "Stunted plant growth",
            "Reduced fruit production"
        ],
        "immediate_actions": [
            "Remove severely infected plants where appropriate",
            "Inspect plants for whiteflies",
            "Separate affected plants when practical"
        ],
        "prevention": [
            "Control whitefly populations",
            "Use healthy planting material",
            "Remove weed hosts around production areas",
            "Use suitable resistant varieties when available"
        ],
        "treatment": [
            "There is no direct cure that eliminates the virus from an infected plant",
            "Management should focus on vector control and removal of infected plants"
        ]
    },

    "Tomato Mosaic Virus": {
        "description": "A viral disease that can cause mosaic patterns, leaf distortion and reduced plant growth.",
        "symptoms": [
            "Light and dark green mosaic patterns",
            "Leaf distortion",
            "Reduced plant growth",
            "Reduced fruit quality or production"
        ],
        "immediate_actions": [
            "Remove severely infected plants when appropriate",
            "Avoid handling healthy plants immediately after infected plants",
            "Disinfect tools"
        ],
        "prevention": [
            "Use disease-free seeds and planting material",
            "Wash hands after handling infected plants",
            "Sanitize tools and equipment",
            "Control weeds that may act as hosts"
        ],
        "treatment": [
            "There is no direct cure for an infected plant",
            "Focus on sanitation and preventing spread"
        ]
    },

    "Healthy": {
        "description": "No major tomato leaf disease was detected by the model.",
        "symptoms": [
            "No significant disease symptoms detected"
        ],
        "immediate_actions": [
            "Continue regular plant monitoring",
            "Maintain good plant nutrition and watering practices"
        ],
        "prevention": [
            "Maintain good garden hygiene",
            "Provide adequate spacing and air circulation",
            "Monitor plants regularly"
        ],
        "treatment": [
            "No disease treatment is indicated based on this prediction"
        ]
    }
}

def generate_ai_advice(disease, confidence, advice):
    prompt = f"""
You are CropGuard AI, an agricultural assistant.

A computer vision model analyzed a tomato leaf.

Disease prediction:
{disease}

Model confidence:
{confidence}%

Use ONLY the structured agricultural information provided below.
Do not invent pesticides, chemicals, dosages, application rates,
or unsupported treatments.

STRUCTURED KNOWLEDGE:
{advice}

Give concise, practical advice for a tomato farmer.

Use these sections:

1. What this means
2. What to do now
3. Prevention
4. Treatment considerations
5. Important caution

Use simple language.
If treatment depends on local regulations or product labels,
clearly say so.
"""

    try:
        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return response.output_text

    except Exception as e:
        print(f"Gemini API error: {e}")

        return (
            "AI-generated management advice is temporarily unavailable. "
            "Please use the structured agricultural guidance provided above "
            "and consult a qualified agricultural professional for treatment decisions."
        )



@app.get("/test-advisor")
async def test_advisor():
    disease = "Early Blight"
    confidence = 85.0
    advice = DISEASE_ADVICE[disease]

    ai_advice = generate_ai_advice(
        disease,
        confidence,
        advice
    )

    return {
        "disease": disease,
        "confidence": confidence,
        "ai_advice": ai_advice
    }

# ------------------------------------------------------------
# Home page
# ------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "ResNet50",
        "classes": len(CLASS_NAMES)
    }


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    # Open image and convert to RGB
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Run prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Find predicted class
    predicted_class = int(
        np.argmax(predictions[0])
    )

    # Get confidence
    confidence = float(
        predictions[0][predicted_class]
    )

    disease = CLASS_NAMES[predicted_class]
    confidence_percent = round(confidence * 100, 2)

    # --------------------------------------------------------
    # Low-confidence safeguard
    # --------------------------------------------------------

    if confidence_percent < 60:
        return {
            "disease": "Uncertain",
            "confidence": confidence_percent,
            "advice": {
                "description": "The model is not sufficiently confident in this prediction.",
                "symptoms": [],
                "immediate_actions": [
                    "Upload a clearer image of the tomato leaf",
                    "Make sure the affected area is clearly visible",
                    "Avoid making treatment decisions based on this prediction"
                ],
                "prevention": [
                    "Continue monitoring the plant",
                    "Check nearby plants for similar symptoms"
                ],
                "treatment": [
                    "No disease-specific treatment is recommended from this prediction",
                    "Consider confirmation from a qualified agricultural professional"
                ]
            },
            "ai_advice": None
        }

    # --------------------------------------------------------
    # Get structured disease knowledge
    # --------------------------------------------------------

    advice = DISEASE_ADVICE[disease]

    # --------------------------------------------------------
    # Generate Gemini AI advice
    # --------------------------------------------------------

    ai_advice = generate_ai_advice(
        disease,
        confidence_percent,
        advice
    )

    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return {
        "disease": disease,
        "confidence": confidence_percent,
        "advice": advice,
        "ai_advice": ai_advice
    }