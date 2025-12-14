import streamlit as st
from datetime import datetime
import io
import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels
import base64

st.set_page_config(layout="centered")

MODEL_PATHS = {
    "MODEL A": r"Models\Model_A.pth",
    "MODEL B": r"Models\Model_B.pth"
}
ANNOTATION_PATHS = {
    "MODEL A": r"Annotations\Model_A_Annotation.json",
    "MODEL B": r"Annotations\Model_B_Annotation.json",
}

ICON_FOLDER = r"Icons"

ICON_MAPPING = {
    "Chitato_Lite_Rumput_Laut": "Chitato_Lite_Rumput_Laut",
    "Chitato_Chijeu": "Chitato_Chijeu",
    "Chitato_Lite_Salmon_teriyaki": "Chitato_Lite_Salmon_teriyaki",
    "Chitato_Lite_Seoul_Baechu_Kimchi": "Chitato_Lite_Seoul_Baechu_Kimchi",
    "Chitato_Lite_Sour_Cream": "Chitato_Lite_Sour_Cream",
    "Chitato_Rasa_Asli": "Chitato_Rasa_Asli",
    "Chitato_Rose_Tteobokki": "Chitato_Rose_Tteobokki",
    "Chitato_Sapi_Bumbu_Bakar": "Chitato_Sapi_Bumbu_Bakar",
    "Chitato_Sapi_Panggang": "Chitato_Sapi_Panggang",
    "French_Fries_2000": "French_Fries_2000",
    "Chitato_Rumput_Laut_Aburi": "Chitato_Rumput_Laut_Aburi",
    "Chiki_twist": "Chiki_twist"
}

DISPLAY_NAME_MAPPING = {
    "Chiki_twist": "Chiki Twist",
    "Chitato_Chijeu": "Chitato Rasa Chijeu",
    "Chitato_Lite_Rumput_Laut": "Chitato Lite Rasa Rumput Laut",
    "Chitato_Rumput_Laut_Aburi": "Chitato Lite Rasa Rumput Laut Aburi",
    "Chitato_Lite_Salmon_teriyaki": "Chitato Lite Rasa Salmon Teriyaki",
    "Chitato_Lite_Seoul_Baechu_Kimchi": "Chitato Lite Rasa Kimchi",
    "Chitato_Lite_Sour_Cream": "Chitato Lite Rasa Sour Cream",
    "Chitato_Rasa_Asli": "Chitato Rasa Asli",
    "Chitato_Rose_Tteobokki": "Chitato Rasa Rose Tteobokki",
    "Chitato_Sapi_Bumbu_Bakar": "Chitato Rasa Sapi Bumbu Bakar",
    "Chitato_Sapi_Panggang": "Chitato Rasa Sapi Panggang",
    "French_Fries_2000": "French Fries 2000"
}

if 'page' not in st.session_state:
    st.session_state.page = 'CAMERA'
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_history_item' not in st.session_state:
    st.session_state.selected_history_item = None
if 'came_from' not in st.session_state:
    st.session_state.came_from = 'CAMERA'
if 'current_batch_results' not in st.session_state:
    st.session_state.current_batch_results = []
if 'current_batch_index' not in st.session_state:
    st.session_state.current_batch_index = 0
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'MODEL A'

def get_class_mappings(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    if 'categories' in data:
        idx_to_class = {cat['id']: cat['name'] for cat in data['categories']}
    else:
        all_labels = set()
        for item in data:
            for ann in item.get("label", []):
                if ann.get("rectanglelabels"):
                    all_labels.add(ann["rectanglelabels"][0])
        class_to_idx = {label: i + 1 for i, label in enumerate(sorted(list(all_labels)))}
        idx_to_class = {i: label for label, i in class_to_idx.items()}

    idx_to_class[0] = 'background'
    return idx_to_class

def load_inference_model(num_classes, model_path, device):
    print("Loading model architecture...")
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    )

    print("Replacing classification head")
    out_channels = retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    new_head = SSDLiteClassificationHead(
        in_channels=out_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=torch.nn.BatchNorm2d
    )
    model.head.classification_head = new_head

    print(f"Getting model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("Model loaded.")
    return model

@st.cache_resource
def load_model_and_dependencies(model_name):
    
    MODEL_PATH = MODEL_PATHS[model_name]
    ANNOTATION_PATH = ANNOTATION_PATHS[model_name]
    
    if not os.path.exists(MODEL_PATH):
        error_msg = f"Model file not found in {MODEL_PATH}"
        print(error_msg)
        return None, None, None, None, error_msg
        
    if not os.path.exists(ANNOTATION_PATH):
        error_msg = f"Annotation file not found in {ANNOTATION_PATH}"
        print(error_msg)
        return None, None, None, None, error_msg 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device}")
    
    idx_to_class = get_class_mappings(ANNOTATION_PATH)
    num_classes = len(idx_to_class)
    print(f"Class amount: {num_classes} (including background)")
    
    model = load_inference_model(num_classes, MODEL_PATH, device)
    
    all_products = set(name for name in idx_to_class.values() if name != 'background')
    
    return model, idx_to_class, device, all_products, None

def preprocess_image(image_buffer, target_size=320):

    image = Image.open(image_buffer).convert("RGB")
    original_w, original_h = image.size
    original_size = (original_w, original_h)

    if original_w == original_h:
        print("Image is 1:1. Resizing directly.")
        cropped_image = image  
        
        left_offset = 0
        top_offset = 0
        
        resize_transform = T.Resize((target_size, target_size), interpolation=T.InterpolationMode.LANCZOS)
        resized_image_320 = resize_transform(cropped_image)
        
        scale_ratio = original_w / target_size 

    else:
        print("Image is not 1:1. Center cropping.")
        min_dim = min(original_w, original_h)
        
        left_offset = (original_w - min_dim) // 2
        top_offset = (original_h - min_dim) // 2
        
        crop_transform = T.CenterCrop(min_dim)
        cropped_image = crop_transform(image)
        
        resize_transform = T.Resize((target_size, target_size), interpolation=T.InterpolationMode.LANCZOS)
        resized_image_320 = resize_transform(cropped_image)
        
        scale_ratio = min_dim / target_size

    # Scaling info kept for potential future use, though not used for drawing anymore
    scaling_info = {
        "left_offset": left_offset,
        "top_offset": top_offset,
        "scale_ratio": scale_ratio
    }

    return resized_image_320, cropped_image, original_size, scaling_info

def run_detection(image_buffer):
    resized_image_320, cropped_image_1to1, original_size, scaling_info = preprocess_image(image_buffer, target_size=320)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    
    transform = T.Compose([
        T.ToTensor(),
        normalize
    ])
    
    image_tensor = transform(resized_image_320).unsqueeze(0).to(device)

    print("Running inference...")
    with torch.no_grad():
        predictions = model(image_tensor)

    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels_idx = predictions[0]['labels'].cpu().numpy()

    # Filter based on threshold
    keep_indices = pred_scores >= SCORE_THRESHOLD
    
    filtered_labels_idx = pred_labels_idx[keep_indices]
    filtered_scores = pred_scores[keep_indices]
    
    # Map indices to names
    filtered_labels = [idx_to_class[i] for i in filtered_labels_idx]
    
    # Prepare In-Stock list
    hasil_in_stock = list(zip(filtered_labels, filtered_scores))
    hasil_in_stock.sort(key=lambda x: (-x[1], DISPLAY_NAME_MAPPING.get(x[0], x[0])))
    
    # Prepare OOS list
    detected_products = set(filtered_labels) 
    not_detected_products = ALL_PRODUCTS - detected_products
    
    hasil_oos = []
    
    sorted_oos_internal_names = sorted(
        list(not_detected_products), 
        key=lambda name: DISPLAY_NAME_MAPPING.get(name, name)
    )
    
    for product in sorted_oos_internal_names:
        hasil_oos.append((product, 0))

    # Convert the clean cropped image to bytes for display (NO bounding boxes)
    buf = io.BytesIO()
    cropped_image_1to1.save(buf, format='PNG')
    buf.seek(0)

    return {
        'in_stock': hasil_in_stock,
        'oos': hasil_oos,
        'processed_image': buf
    }

ABOUT_APP = """This application will detect 12 types of snack products which are:
- Chiki Twist
- Chitato Rasa Asli
- Chitato Rasa Chijeu
- Chitato Rasa Rose Tteobokki
- Chitato Rasa Sapi Bumbu Bakar
- Chitato Rasa Sapi Panggang
- Chitato Lite Rasa Rumput Laut
- Chitato Lite Rasa Rumput Laut Aburi
- Chitato Lite Rasa Sour Cream
- Chitato Lite Rasa Salmon Teriyaki
- Chitato Lite Rasa Kimchi
- French Fries 2000

To use this web application, users can follow these steps:
- Take a photo via webcam with the ‘Camera’ module or insert an image in the ‘Files’ module.
- Select Model A or Model B.
- Perform detection with ‘Start Detection’.
- View the results.
- To view previous results, users can go to the ‘History’ module.

Users can use 2 types of MobileNetV3-Large models such as:
- Model A with the following specifications:
    - Number of Training Datasets per class: 134 images with a total of 1,608 images
    - Number of Validation Datasets per class: 34 images with a total of 408 images
    - 100 Epoch
    - Batch Size 16
    - Specializes in images taken with an angled perspective
- Model B with the following specifications:
    - Number of training datasets per class: 168 images, with a total of 2,016 images
    - Number of validation datasets per class: 42 images, with a total of 504 images
    - 100 epochs
    - Batch size: 16
    - Specializes in images taken with a frontal perspective
    """
ABOUT_DEVELOPER = "Christabella Jocelynne Chandra - 535220166"


import base64 

@st.cache_data
def _get_image_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def display_result_data(result):
    # Display the clean image
    st.image(result['image'], width='stretch')
    st.caption(f"Detection done on {result['date']}")
    
    st.write("") 
    
    col1, col2 = st.columns(2) 
    
    with col1:
        st.subheader("IN STOCK (Detected)")

        in_stock_names = []
        for item, score in result['in_stock']:
            # Format skor
            if 0.001 <= score <= 1000:
                score_str = f"{score:.6f}" 
            else:
                score_str = f"{score:.2e}"  
            in_stock_names.append(f"- {DISPLAY_NAME_MAPPING.get(item, item)} ({score_str})")

        in_stock_string = "\n".join(in_stock_names)

        if in_stock_string: 
            with st.expander("Copy IN STOCK List"):
                st.text_area(
                    "in_stock_list_copy", 
                    value=in_stock_string, 
                    height=150,
                    label_visibility="collapsed"
                )

        with st.container(border=True): 
            if not result['in_stock']:
                st.write("No products detected.")
            else:
                for item, score in result['in_stock']: 
                    icon_filename = ICON_MAPPING.get(item, "default")
                    icon_path = os.path.join(ICON_FOLDER, f"{icon_filename}.png")
                    display_name = DISPLAY_NAME_MAPPING.get(item, item)

                    # Format skor
                    if 0.001 <= score <= 1000:
                        score_str = f"{score:.6f}"
                    else:
                        score_str = f"{score:.2e}"
                    
                    if os.path.exists(icon_path):
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <img src="data:image/png;base64,{_get_image_as_base64(icon_path)}" 
                                     alt="{item}" style="width: 24px; height: 24px; margin-right: 8px;">
                                <p style="margin: 0; font-size: 16px;">{display_name} ({score_str})</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.write(f"{display_name} ({score_str})")
    
    with col2:
        st.subheader("OOS (Undetected)")

        oos_names = [f"- {DISPLAY_NAME_MAPPING.get(item, item)} (0)" for item, count in result['oos']]
        oos_string = "\n".join(oos_names)

        if oos_string:
            with st.expander("Copy OOS List"):
                st.text_area(
                    "oos_list_copy",
                    value=oos_string, 
                    height=150,
                    label_visibility="collapsed"
                )

        with st.container(border=True, height=400): 
            if not result['oos']:
                st.write("All products detected.")
            else:
                for item, count in result['oos']: 
                    icon_filename = ICON_MAPPING.get(item, "default")
                    icon_path = os.path.join(ICON_FOLDER, f"{icon_filename}.png")
                    display_name = DISPLAY_NAME_MAPPING.get(item, item)
                    
                    if os.path.exists(icon_path):
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <img src="data:image/png;base64,{_get_image_as_base64(icon_path)}" 
                                     alt="{item}" style="width: 24px; height: 24px; margin-right: 8px;">
                                <p style="margin: 0; font-size: 16px;">{display_name} (0)</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.write(f"{display_name} (0)")
                    
def render_camera_page():
    col_title, col_help = st.columns([0.85, 0.15])
    with col_title:
        st.title("CAMERA")
    with col_help:
        if st.button("Help", width='stretch'):
            st.session_state.came_from = 'CAMERA'
            st.session_state.page = 'HELP'
            st.rerun()

    img_file_buffer = st.camera_input("Point webcam to shelf", label_visibility="collapsed")

    if img_file_buffer:
        if st.button("Start Detection", width='stretch', type="primary"):
            with st.spinner("Running detection..."):
                results = run_detection(img_file_buffer)
                current_date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                
                new_result = {
                    'date': current_date_time,
                    'image': results['processed_image'], 
                    'in_stock': results['in_stock'],
                    'oos': results['oos']
                }
                
                st.session_state.current_batch_results = [new_result]
                st.session_state.current_batch_index = 0
                
                st.session_state.history.insert(0, new_result) 
                st.session_state.came_from = 'CAMERA'
                st.session_state.page = 'RESULTS'
                st.rerun()

def render_files_page():
    col_title, col_help = st.columns([0.85, 0.15])
    with col_title:
        st.title("FILES")
    with col_help:
        if st.button("Help", width='stretch'):
            st.session_state.came_from = 'CAMERA'
            st.session_state.page = 'HELP'
            st.rerun()

    uploaded_files = st.file_uploader(
        "Upload image", 
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        
        files_to_process = uploaded_files
        if len(uploaded_files) > 10:
            st.warning(f"You uploaded {len(uploaded_files)} files. Only the first 10 files will be processed.")
            files_to_process = uploaded_files[:10]
        
        with st.expander(f"Showing {len(files_to_process)} images"):
            st.image(files_to_process, caption=[f"Image {i+1}" for i in range(len(files_to_process))])

        if st.button(f"Starting detection on ({len(files_to_process)} files)", width='stretch', type="primary"):
            
            with st.spinner(f"Processing {len(files_to_process)} images..."):
                
                progress_bar = st.progress(0, text="Starting...")
                
                batch_results = []
                
                for i, uploaded_file in enumerate(files_to_process):
                    
                    progress_text = f"Processing image no {i+1}/{len(files_to_process)} ({uploaded_file.name})..."
                    progress_bar.progress((i) / len(files_to_process), text=progress_text)
                    
                    uploaded_file.seek(0)
                    
                    results = run_detection(uploaded_file)
                    current_date_time = datetime.now().strftime(f"%d/%m/%Y %H:%M:%S") + f" (File: {uploaded_file.name})"
                    
                    new_result = {
                        'date': current_date_time,
                        'image': results['processed_image'],
                        'in_stock': results['in_stock'],
                        'oos': results['oos']
                    }
                    
                    batch_results.append(new_result)
                    
                st.session_state.current_batch_results = batch_results
                st.session_state.current_batch_index = 0
                
                st.session_state.history = batch_results[::-1] + st.session_state.history
                
                progress_bar.progress(1.0, text="Finished!")

                st.session_state.page = 'RESULTS'
                st.rerun()

def render_results_page():
    col_title, col_help = st.columns([0.85, 0.15])
    with col_title:
        st.title("RESULTS")
    with col_help:
        if st.button("Help", width='stretch'):
            st.session_state.came_from = 'CAMERA'
            st.session_state.page = 'HELP'
            st.rerun()
    
    if not st.session_state.get('current_batch_results'):
        st.warning("There is no previous detection results")
        if st.button("Return"):
            st.session_state.page = 'RESULTS'
            st.rerun()
        return

    batch = st.session_state.current_batch_results
    index = st.session_state.current_batch_index
    result = batch[index]
    is_batch = len(batch) > 1

    if is_batch:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("←", width='stretch', disabled=(index == 0)):
                st.session_state.current_batch_index -= 1
                st.rerun()
        with col2:
            st.markdown(f"<h5 style='text-align: center; margin-top: 10px;'>Result {index + 1} / {len(batch)}</h5>", unsafe_allow_html=True)
        with col3:
            if st.button("→", width='stretch', disabled=(index == len(batch) - 1)):
                st.session_state.current_batch_index += 1
                st.rerun()
        st.divider()

    st.write("")

    display_result_data(result)

    st.divider()
    if st.button("FINISH", width='stretch'):
        st.session_state.current_batch_results = []
        st.session_state.current_batch_index = 0
        return_page = st.session_state.get('came_from', 'CAMERA') 
        st.session_state.page = return_page
        st.rerun()

def render_history_page():
    col_title, col_help = st.columns([0.85, 0.15])
    with col_title:
        st.title("HISTORY")
    with col_help:
        if st.button("Help", width='stretch'):
            st.session_state.came_from = 'HISTORY'
            st.session_state.page = 'HELP'
            st.rerun()

    if not st.session_state.history:
        st.info("There is no previous detection history found.")
        return

    if st.session_state.selected_history_item:
        st.header(f"Results for {st.session_state.selected_history_item['date']}")
        

        st.write("")

        display_result_data(st.session_state.selected_history_item)

        st.divider()
        if st.button("FINISH", use_container_width=True): 
            st.session_state.selected_history_item = None
            st.session_state.current_batch_results = []
            st.session_state.current_batch_index = 0
            st.rerun() 
    else:
        st.subheader("Detection History")
        st.caption("Sorted from most recent.")
        
        if st.button("Delete History", type="secondary", width='stretch'):
            st.session_state.history = []
            st.rerun()
        
        st.divider()
        
        for item in st.session_state.history:
            date_str = item['date']
            if st.button(date_str, width='stretch'):
                st.session_state.selected_history_item = item
                st.rerun()

def render_help_page():
    st.title("ABOUT")
    origin_page = st.session_state.get('came_from', 'CAMERA')
    st.subheader("ABOUT APP")
    st.write(ABOUT_APP)
    st.subheader("ABOUT DEVELOPER")
    st.write(ABOUT_DEVELOPER)
    st.divider()
    if st.button("FINISH", use_container_width=True): 
        st.session_state.page = origin_page 
        st.rerun()

with st.sidebar:
    st.title("NAVIGASI")
    if st.button("CAMERA", width='stretch'):
        st.session_state.page = 'CAMERA'
        st.session_state.current_batch_results = []
        st.session_state.current_batch_index = 0
        st.rerun()
    if st.button("FILES", width='stretch'):
        st.session_state.page = 'FILES'
        st.session_state.current_batch_results = []
        st.session_state.current_batch_index = 0
        st.rerun()
    if st.button("HISTORY", width='stretch'):
        st.session_state.page = 'HISTORY'
        st.session_state.selected_history_item = None
        st.session_state.current_batch_results = []
        st.session_state.current_batch_index = 0
        st.rerun()

    st.divider()
    st.title("PILIH MODEL")
    
    st.radio(
        "Choose Model Type",
        options=['MODEL A', 'MODEL B'],
        key='selected_model', 
        label_visibility="collapsed"
    )

SCORE_THRESHOLD = 0.001

placeholder = st.empty()

with placeholder.container():
    st.markdown(
        """
        <style>
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .fade-in {
            animation: fadeIn 1.5s ease-in-out;
        }
        </style>

        <div class="fade-in">
            <h1 style="text-align:center; font-size:50px; margin-top: 100px;">
                SHELFSCAN
            </h1>
            <p style="text-align:center; font-size:20px;">
                Loading intelligent shelf detection model...
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    model, idx_to_class, device, ALL_PRODUCTS, error_message = load_model_and_dependencies(
        st.session_state.selected_model
    )    

placeholder.empty()


if error_message:
    st.error(error_message)
    st.stop()

if model is None:
    st.error("Model can't be loaded")
    st.stop()

if st.session_state.page == 'CAMERA':
    render_camera_page()
elif st.session_state.page == 'FILES':
    render_files_page()
elif st.session_state.page == 'RESULTS':
    render_results_page()
elif st.session_state.page == 'HISTORY':
    render_history_page()
elif st.session_state.page == 'HELP':
    render_help_page()