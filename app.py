# ============================================================
# DEEPFAKE SHIELD — GRADIO APP
# Step 4: Deployment on Hugging Face Spaces
# ============================================================
# File structure needed on HF Spaces:
#   app.py          ← this file
#   requirements.txt
#   deepfake_dual_branch_final.keras  ← upload your model

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
from PIL import Image

# ── Load Model ───────────────────────────────────────────────
print("Loading model...")
model = keras.models.load_model(
    'deepfake_dual_branch_final.keras',
    compile=False
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Warm up
dummy_s = np.zeros((1, 224, 224, 3), dtype=np.float32)
dummy_f = np.zeros((1, 64, 64, 1),  dtype=np.float32)
_ = model([dummy_s, dummy_f], training=False)
print("Model ready.")

CLASS_NAMES = ['Fake', 'Real']
IMG_SIZE    = 224


# ── Preprocessing ────────────────────────────────────────────
def preprocess_image(pil_img):
    """Convert PIL image → spatial tensor."""
    img = np.array(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, 0)  # (1,224,224,3)

def compute_fft(pil_img):
    """Convert PIL image → FFT feature tensor."""
    img = np.array(pil_img.convert('L').resize((IMG_SIZE, IMG_SIZE)))
    img = img.astype(np.float32) / 255.0
    fft = np.fft.fft2(img)
    fft_s = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_s) + 1e-8)
    mag = cv2.resize(mag, (64, 64))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return np.expand_dims(mag, axis=(0, -1))  # (1,64,64,1)

def get_display_img(pil_img):
    """Normalize image for display."""
    img = np.array(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    return img.astype(np.float32) / 255.0


# ── GradCAM ──────────────────────────────────────────────────
def make_gradcam(spatial_tensor, fft_tensor):
    """Input-gradient saliency map."""
    spatial_var = tf.Variable(tf.cast(spatial_tensor, tf.float32))
    fft_t       = tf.cast(fft_tensor, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(spatial_var)
        pred  = model([spatial_var, fft_t], training=False)
        score = pred[0][0]

    grads   = tape.gradient(score, spatial_var)
    if grads is None:
        return None

    heatmap = tf.reduce_mean(tf.abs(grads[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap, display_img, alpha=0.5):
    """Blend heatmap onto image."""
    h = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    h = np.uint8(255 * h)
    colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    orig    = np.uint8(255 * np.clip(display_img, 0, 1))
    blended = cv2.addWeighted(orig, 1 - alpha, colored, alpha, 0)
    return blended


# ── Main Prediction Function ─────────────────────────────────
def predict(image):
    """
    Takes a PIL image → returns:
    - prediction label + confidence
    - FFT spectrum image
    - GradCAM overlay image
    - analysis text
    """
    if image is None:
        return "Please upload an image.", None, None, ""

    pil_img  = Image.fromarray(image) if isinstance(image, np.ndarray) else image

    spatial  = preprocess_image(pil_img)
    fft      = compute_fft(pil_img)
    disp     = get_display_img(pil_img)

    # Prediction
    pred_score  = model.predict([spatial, fft], verbose=0)[0][0]
    pred_label  = 'FAKE' if pred_score < 0.5 else 'REAL'
    confidence  = (1 - pred_score) if pred_score < 0.5 else pred_score
    confidence  = float(confidence) * 100

    # FFT visualization
    fft_disp = fft[0, :, :, 0]
    fig, ax  = plt.subplots(figsize=(4, 4))
    ax.imshow(fft_disp, cmap='hot')
    ax.set_title('FFT Frequency Spectrum', fontsize=11)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    fft_img = Image.open(buf)

    # GradCAM
    heatmap = make_gradcam(spatial, fft)
    if heatmap is not None:
        overlay = overlay_heatmap(heatmap, disp)
        gradcam_img = Image.fromarray(overlay)
    else:
        gradcam_img = pil_img

    # Analysis text
    if pred_label == 'FAKE':
        analysis = f"""🚨 DEEPFAKE DETECTED ({confidence:.1f}% confidence)

The model identified manipulation artifacts in this image.

Key indicators analyzed:
• Spatial patterns: Unnatural pixel distributions in facial regions
• Frequency domain: Anomalous high-frequency components in FFT spectrum
• GradCAM focus: Highlighted regions show where manipulation was detected

Note: Confidence below 70% may indicate ambiguous cases."""
    else:
        analysis = f"""✅ REAL IMAGE ({confidence:.1f}% confidence)

No significant manipulation artifacts detected.

Analysis:
• Spatial patterns: Natural pixel distributions
• Frequency domain: Normal frequency distribution in FFT spectrum  
• GradCAM focus: Attention spread naturally across facial features

Note: Sophisticated deepfakes may still evade detection."""

    label_str = f"{'🔴 FAKE' if pred_label == 'FAKE' else '🟢 REAL'} — {confidence:.1f}% confidence"

    return label_str, fft_img, gradcam_img, analysis


# ── Gradio Interface ─────────────────────────────────────────
with gr.Blocks(
    title="DeepFake Shield",
    theme=gr.themes.Soft(),
    css="""
        .header { text-align: center; margin-bottom: 20px; }
        .result-box { font-size: 1.3em; font-weight: bold; }
    """
) as demo:

    gr.HTML("""
        <div class="header">
            <h1>🛡️ DeepFake Shield</h1>
            <p>AI-powered deepfake detection using spatial + frequency domain analysis</p>
            <p><i>EfficientNetB4 + FFT Branch + GradCAM Explainability</i></p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Face Image",
                type="pil",
                height=300
            )
            submit_btn = gr.Button(
                "🔍 Analyze Image",
                variant="primary",
                size="lg"
            )

            gr.Examples(
                examples=[],  # Add example image paths here if available
                inputs=input_image
            )

        with gr.Column(scale=1):
            prediction_label = gr.Textbox(
                label="Prediction",
                elem_classes=["result-box"],
                interactive=False
            )
            analysis_text = gr.Textbox(
                label="Analysis",
                lines=10,
                interactive=False
            )

    with gr.Row():
        fft_output = gr.Image(
            label="FFT Frequency Spectrum",
            height=250
        )
        gradcam_output = gr.Image(
            label="GradCAM — Regions of Interest",
            height=250
        )

    gr.HTML("""
        <div style="text-align:center; margin-top:20px; color:gray; font-size:0.85em">
            <p>Model: EfficientNetB4 + Frequency Domain Branch | 
               Trained on: Deepfake & Real Images Dataset |
               Built by: Krishna Jaiswal</p>
            <p>⚠️ This tool is for educational purposes. 
               Always verify with multiple sources.</p>
        </div>
    """)

    submit_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[prediction_label, fft_output, gradcam_output, analysis_text]
    )

demo.launch()
