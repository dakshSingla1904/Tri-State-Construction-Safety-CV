import cv2
import time
import math
import numpy as np
import os
import threading
import requests
import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Forces Matplotlib to run in headless/thread-safe mode
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import warnings

# --- IMPORTS FOR GMAIL (PURE SMTP RPA) ---
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

warnings.filterwarnings('ignore')

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
EXPORT_FOLDER = os.path.join(BASE_DIR, 'exports')

INCIDENTS_FOLDER = os.path.join(EXPORT_FOLDER, 'incidents')
REPORTS_FOLDER = os.path.join(EXPORT_FOLDER, 'reports')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)
os.makedirs(INCIDENTS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

class SystemState:
    def __init__(self):
        self.mode = "unified" 
        self.input_type = "image"
        self.file_path = ""        
        self.w_hat = 20
        self.w_vest = 15
        self.w_mask = 5  
        self.w_ergo = 15
        self.telemetry = {"score": 100, "workers": 0, "hats": 0, "vests": 0, "masks": 0, "ergo": 0, "falls": 0, "fps": 0.0}
        
        # --- NATIVE EMAIL SMTP CREDENTIALS ---
        self.gmail_user = "YOUR_EMAIL@gmail.com" 
        self.gmail_app_password = "YOUR_APP_PASSWORD" 
        self.boss_email = "MANAGER_EMAIL@gmail.com"
        
        # --- INDEPENDENT COOLDOWNS ---
        self.last_critical_time = 0 
        self.last_warning_time = 0
        
        self.is_recording = True
        self.stream_id = 0.0 
        self.writer_normal = None
        self.writer_slow = None
        self.last_saved_frame = None

        self.bg_status = "idle"
        self.bg_current_frame = 0
        self.bg_total_frames = 1
        self.bg_eta = "00:00"
        self.bg_message = ""

        self.shift_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(REPORTS_FOLDER, f"Shift_Log_{self.shift_id}.csv")
        self.worker_violations = {} 
        self.last_logged_time = {}  
        
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Worker_ID", "Violation_Type", "System_Score"])

STATE = SystemState()

class ModelManager:
    def __init__(self):
        self.scale = "medium"
        self.m1 = None
        self.m2 = None
        self.m_pose = None
        
        self.path_m1_med = "weights/m1_medium.pt"
        self.path_m2_med = "weights/m2_medium.pt"
        self.path_small = "weights/m_small.pt"
        
        self.load_models("medium")
        
    def load_models(self, scale):
        print(f"\n-> Loading {scale.upper()} Neural Networks into Memory...")
        self.scale = scale
        
        if scale == "small":
            self.m1 = YOLO(self.path_small)
            self.m2 = None
        elif scale == "mixed":
            self.m1 = YOLO(self.path_small)
            self.m2 = YOLO(self.path_m2_med)
        else:
            self.m1 = YOLO(self.path_m1_med)
            self.m2 = YOLO(self.path_m2_med)
            
        self.m_pose = YOLO("yolo11n-pose.pt")
        print("-> Models loaded successfully!")

MODELS = ModelManager()

ID_MAP_MED = {0: 1, 1: 2, 5: 0, 6: 3, 7: 4, 8: 5} 
EXPERT_MATRIX_MED = {0: (1.2, 0.8), 1: (0.8, 1.2), 2: (0.7, 1.3), 3: (1.3, 0.7), 4: (1.0, 1.0), 5: (1.0, 1.0)}

ID_MAP_SMALL = {5: 0, 0: 1, 1: 2, 7: 3, 8: 4, 10: 5} 
NEG_MAP_SMALL = {2: 2, 3: 3, 4: 4} 

HYBRID_MATRIX = {
    0: (1.0, 1.0),  
    1: (0.8, 1.2),  
    2: (0.2, 1.8),  
    3: (1.0, 1.0),  
    4: (1.3, 0.7),  
    5: (1.0, 1.0)   
}

SKELETON_PAIRS = [(5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

def fire_rpa_alert(tier, message, data=None):
    current_time = time.time()
    
    if tier == "CRITICAL":
        if current_time - STATE.last_critical_time < 15: return
        STATE.last_critical_time = current_time
    elif tier == "WARNING":
        if current_time - STATE.last_warning_time < 30: return 
        STATE.last_warning_time = current_time
    
    def execute_routing():
        try:
            if not STATE.gmail_user or not STATE.gmail_app_password or not STATE.boss_email:
                print(f"\n[RPA LOG] {tier} Alert: {message} (Email skipped: Credentials missing)")
                return

            msg = MIMEMultipart()
            msg['From'] = STATE.gmail_user
            msg['To'] = STATE.boss_email
            
            if tier == "CRITICAL":
                msg['Subject'] = f"🚨 URGENT: Life-Safety Alert (Camera 1)"
                body = f"CRITICAL INCIDENT DETECTED\n\nLocation: Camera 1 (Edge Node)\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nDetails:\n{message}\n\nPhysical Sirens have been activated. Action Required Immediately."
            
            elif tier == "WARNING":
                msg['Subject'] = f"⚠️ WARNING: Compliance Threshold Exceeded"
                body = f"SAFETY PROTOCOL WARNING\n\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nDetails:\n{message}\n\nPlease dispatch a supervisor to the zone."
            
            elif tier == "ADMINISTRATIVE":
                msg['Subject'] = f"📊 Automated Safety Audit - {datetime.now().strftime('%Y-%m-%d')}"
                body = f"Attached is the automated safety digest for shift: {STATE.shift_id}.\n\n{message}"
                
                pdf_path = data.get("report") if data else None
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        attach = MIMEApplication(f.read(), _subtype="pdf")
                        attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
                        msg.attach(attach)

            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(STATE.gmail_user, STATE.gmail_app_password)
            server.send_message(msg)
            server.quit()
            print(f"\n[RPA SUCCESS] {tier} Email securely delivered to {STATE.boss_email}!")

        except Exception as e:
            print(f"\n[RPA ERROR] SMTP Routing failed for {tier}: {e}")
            
    threading.Thread(target=execute_routing, daemon=True).start()

def append_dashboard(frame, telemetry, mode_name):
    h, w = frame.shape[:2]
    
    dash_w = max(340, int(w * 0.35))
    dash = np.full((h, dash_w, 3), (35, 25, 20), dtype=np.uint8) 
    
    cv2.rectangle(dash, (2, 2), (dash_w - 2, h - 2), (60, 50, 40), 2)
    cv2.line(dash, (0, 0), (0, h), (200, 200, 200), 3) 
    
    fs_title = dash_w / 500.0
    fs_text = dash_w / 650.0
    fs_score = dash_w / 400.0
    
    y_gap = max(25, int(h * 0.065))
    y = max(35, int(h * 0.08))
    
    cv2.putText(dash, "AI COMMAND CENTER", (15, y), cv2.FONT_HERSHEY_DUPLEX, fs_title, (255, 204, 0), 1)
    y += int(y_gap * 0.8)
    cv2.putText(dash, f"Mode: {mode_name.upper()}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, fs_text, (200,200,200), 1)
    y += int(y_gap * 0.5)
    cv2.line(dash, (15, y), (dash_w - 15, y), (100, 100, 100), 1)
    y += y_gap
    
    cv2.putText(dash, f"Workers: {telemetry['workers']}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, fs_text, (255,255,255), 1)
    y += y_gap
    cv2.putText(dash, f"Missing Hats  : {telemetry['hats']}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, fs_text, (0,0,255) if telemetry['hats']>0 else (0,255,0), 1)
    y += y_gap
    cv2.putText(dash, f"Missing Vests : {telemetry['vests']}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, fs_text, (0,0,255) if telemetry['vests']>0 else (0,255,0), 1)
    y += y_gap
    cv2.putText(dash, f"Missing Masks : {telemetry['masks']}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, fs_text, (0,0,255) if telemetry['masks']>0 else (0,255,0), 1)
    y += y_gap
    cv2.putText(dash, f"Ergo Penalty  : {telemetry['ergo']} pts", (15, y), cv2.FONT_HERSHEY_SIMPLEX, fs_text, (0,165,255) if telemetry['ergo']>0 else (0,255,0), 1)
    
    y += int(y_gap * 1.5)
    if telemetry['falls'] > 0:
        box_h = int(y_gap * 1.2)
        cv2.rectangle(dash, (15, y - box_h), (dash_w - 15, y + int(box_h*0.3)), (0,0,255), -1)
        cv2.putText(dash, "FALL DETECTED", (25, y), cv2.FONT_HERSHEY_DUPLEX, fs_title, (255,255,255), 2)
        
        # --- VIRTUAL HARDWARE SIREN (Flashing Red Border) ---
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
            cv2.putText(frame, "🚨 SIREN ACTIVE 🚨", (int(w*0.5) - 200, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4)

    score_color = (0,255,0) if telemetry['score'] > 80 else (0,165,255) if telemetry['score'] > 50 else (0,0,255)
    cv2.putText(dash, f"SCORE: {telemetry['score']}/100", (15, h - 25), cv2.FONT_HERSHEY_DUPLEX, fs_score, score_color, 2)
    
    return np.hstack((frame, dash))

class BoundingBoxSmoother:
    def __init__(self, max_missing=3, alpha=0.6):
        self.tracks = []; self.max_missing = max_missing; self.alpha = alpha 
    def calculate_iou(self, box1, box2):
        x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
        x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top: return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        return intersection_area / float(((box1[2]-box1[0])*(box1[3]-box1[1])) + ((box2[2]-box2[0])*(box2[3]-box2[1])) - intersection_area)
    def update(self, new_boxes, new_labels):
        matched_new, matched_tracks = set(), set()
        for t_idx, track in enumerate(self.tracks):
            best_iou, best_n_idx = 0, -1
            for n_idx, (nb, nl) in enumerate(zip(new_boxes, new_labels)):
                if n_idx in matched_new or track['label'] != nl: continue
                iou = self.calculate_iou(track['box'], nb)
                if iou > best_iou: best_iou, best_n_idx = iou, n_idx
            if best_iou > 0.3:
                matched_nb = new_boxes[best_n_idx]
                track['box'] = [
                    self.alpha * matched_nb[0] + (1 - self.alpha) * track['box'][0],
                    self.alpha * matched_nb[1] + (1 - self.alpha) * track['box'][1],
                    self.alpha * matched_nb[2] + (1 - self.alpha) * track['box'][2],
                    self.alpha * matched_nb[3] + (1 - self.alpha) * track['box'][3]
                ]
                track['missing'] = 0
                matched_new.add(best_n_idx); matched_tracks.add(t_idx)
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks: track['missing'] += 1
        for n_idx, (nb, nl) in enumerate(zip(new_boxes, new_labels)):
            if n_idx not in matched_new: self.tracks.append({'box': nb, 'label': nl, 'missing': 0})
        self.tracks = [t for t in self.tracks if t['missing'] <= self.max_missing]
        return [t['box'] for t in self.tracks], [t['label'] for t in self.tracks]

def get_angle(p1, p2, p3):
    if p1[0]==0 or p2[0]==0 or p3[0]==0: return 0
    a = math.atan2(p3[1]-p2[1], p3[0]-p2[0]) - math.atan2(p1[1]-p2[1], p1[0]-p2[0])
    ang = abs(a * 180.0 / math.pi)
    return 360 - ang if ang > 180 else ang

# --- FULL REBA ERGONOMICS INTEGRATION ---
def calculate_ergonomics(kps):
    angles = {}
    if len(kps) < 17: return angles
    
    mid_sh = [(kps[5][0] + kps[6][0])/2, (kps[5][1] + kps[6][1])/2] if (kps[5][2]>0.5 and kps[6][2]>0.5) else kps[5][:2]
    mid_hip = [(kps[11][0] + kps[12][0])/2, (kps[11][1] + kps[12][1])/2] if (kps[11][2]>0.5 and kps[12][2]>0.5) else kps[11][:2]
    mid_ear = [(kps[3][0] + kps[4][0])/2, (kps[3][1] + kps[4][1])/2] if (kps[3][2]>0.5 and kps[4][2]>0.5) else kps[3][:2]

    # 1. Trunk Flexion
    if mid_sh[0] != 0 and mid_hip[0] != 0: 
        angles["trunk"] = get_angle(mid_sh, mid_hip, [mid_hip[0], mid_hip[1] - 100])
        
    # 2. Neck Flexion
    if mid_ear[0] != 0 and mid_sh[0] != 0: 
        angles["neck"] = get_angle(mid_ear, mid_sh, [mid_sh[0], mid_sh[1] - 100])
        
    # 3. Shoulder Elevation (Left and Right)
    if kps[5][2] > 0.5 and kps[7][2] > 0.5: 
        angles["shoulder_l"] = get_angle(kps[7][:2], kps[5][:2], [kps[5][0], kps[5][1] + 100])
    if kps[6][2] > 0.5 and kps[8][2] > 0.5: 
        angles["shoulder_r"] = get_angle(kps[8][:2], kps[6][:2], [kps[6][0], kps[6][1] + 100])
        
    # 4. Knee Flexion (Left and Right)
    if kps[11][2] > 0.5 and kps[13][2] > 0.5 and kps[15][2] > 0.5: 
        angles["knee_l"] = get_angle(kps[11][:2], kps[13][:2], kps[15][:2])
    if kps[12][2] > 0.5 and kps[14][2] > 0.5 and kps[16][2] > 0.5: 
        angles["knee_r"] = get_angle(kps[12][:2], kps[14][:2], kps[16][:2])
        
    return angles

def check_ioa(item_box, person_box, is_hat=False):
    px1, py1, px2, py2 = person_box; ex1, ey1, ex2, ey2 = item_box
    if is_hat: py1 = py1 - ((py2 - py1) * 0.25) 
    x_left, y_top = max(px1, ex1), max(py1, ey1)
    x_right, y_bottom = min(px2, ex2), min(py2, ey2)
    if x_right < x_left or y_bottom < y_top: return False 
    return ((x_right - x_left) * (y_bottom - y_top)) / float((ex2 - ex1) * (ey2 - ey1)) > 0.25

def run_smart_ensemble(img):
    scale = MODELS.scale
    inf_size = 640 if scale != "medium" else 1024
    pos_boxes, pos_scores, pos_labels, neg_boxes, neg_labels = [], [], [], [], []

    if scale == "medium":
        res_old = MODELS.m1.predict(img, conf=0.30, imgsz=inf_size, verbose=False)[0]
        res_new = MODELS.m2.predict(img, conf=0.30, imgsz=inf_size, verbose=False)[0]
        
        if len(res_new.boxes) > 0:
            b, s, l = res_new.boxes.xyxyn.cpu().numpy().tolist(), res_new.boxes.conf.cpu().numpy().tolist(), res_new.boxes.cls.cpu().numpy().tolist()
            pos_boxes.append(b); pos_scores.append([min(1.0, s[i] * EXPERT_MATRIX_MED.get(int(l[i]), (1.0,1.0))[1]) for i in range(len(l))]); pos_labels.append([int(x) for x in l])

        if len(res_old.boxes) > 0:
            b, s, l = res_old.boxes.xyxyn.cpu().numpy().tolist(), res_old.boxes.conf.cpu().numpy().tolist(), res_old.boxes.cls.cpu().numpy().tolist()
            cb, cs, cl = [], [], []
            for i, lbl in enumerate(l):
                old_cls = int(lbl)
                if old_cls in [2, 3, 4]: neg_boxes.append(b[i]); neg_labels.append(old_cls)
                else:
                    nid = ID_MAP_MED.get(old_cls, -1)
                    if nid != -1: cb.append(b[i]); cs.append(min(1.0, s[i] * EXPERT_MATRIX_MED.get(nid, (1.0,1.0))[0])); cl.append(nid)
            if cl: pos_boxes.append(cb); pos_scores.append(cs); pos_labels.append(cl)

    elif scale == "mixed":
        res_small = MODELS.m1.predict(img, conf=0.25, imgsz=inf_size, verbose=False)[0]
        res_med = MODELS.m2.predict(img, conf=0.30, imgsz=1024, verbose=False)[0]
        
        if len(res_small.boxes) > 0:
            b, s, l = res_small.boxes.xyxyn.cpu().numpy().tolist(), res_small.boxes.conf.cpu().numpy().tolist(), res_small.boxes.cls.cpu().numpy().tolist()
            cb, cs, cl = [], [], []
            for i, lbl in enumerate(l):
                c_id = int(lbl)
                if c_id in ID_MAP_SMALL: 
                    u_id = ID_MAP_SMALL[c_id]
                    w_small, _ = HYBRID_MATRIX.get(u_id, (1.0, 1.0))
                    cb.append(b[i]); cs.append(min(1.0, s[i] * w_small)); cl.append(u_id)
                elif c_id in NEG_MAP_SMALL: 
                    neg_boxes.append(b[i]); neg_labels.append(NEG_MAP_SMALL[c_id])
            if cl: pos_boxes.append(cb); pos_scores.append(cs); pos_labels.append(cl)
            
        if len(res_med.boxes) > 0:
            b, s, l = res_med.boxes.xyxyn.cpu().numpy().tolist(), res_med.boxes.conf.cpu().numpy().tolist(), res_med.boxes.cls.cpu().numpy().tolist()
            cb, cs, cl = [], [], []
            for i, lbl in enumerate(l):
                u_id = int(lbl)
                _, w_med = HYBRID_MATRIX.get(u_id, (1.0, 1.0))
                cb.append(b[i]); cs.append(min(1.0, s[i] * w_med)); cl.append(u_id)
            if cl: pos_boxes.append(cb); pos_scores.append(cs); pos_labels.append(cl)

    else:
        res_small = MODELS.m1.predict(img, conf=0.25, imgsz=inf_size, verbose=False)[0]
        if len(res_small.boxes) > 0:
            b, s, l = res_small.boxes.xyxyn.cpu().numpy().tolist(), res_small.boxes.conf.cpu().numpy().tolist(), res_small.boxes.cls.cpu().numpy().tolist()
            cb, cs, cl = [], [], []
            for i, lbl in enumerate(l):
                c_id = int(lbl)
                if c_id in ID_MAP_SMALL: cb.append(b[i]); cs.append(s[i]); cl.append(ID_MAP_SMALL[c_id])
                elif c_id in NEG_MAP_SMALL: neg_boxes.append(b[i]); neg_labels.append(NEG_MAP_SMALL[c_id])
            if cl: pos_boxes.append(cb); pos_scores.append(cs); pos_labels.append(cl)

    wbf_boxes, wbf_labels = [], []
    if pos_boxes: wbf_boxes, _, wbf_labels = weighted_boxes_fusion(pos_boxes, pos_scores, pos_labels, weights=[1.0]*len(pos_boxes), iou_thr=0.60, skip_box_thr=0.15)
    return wbf_boxes, wbf_labels, neg_boxes, neg_labels

def process_single_frame(frame, pos_smoother, neg_smoother, temporal_trackers, next_tracker_id, is_image=False):
    h, w = frame.shape[:2]
    wbf_boxes, wbf_labels, neg_boxes, neg_labels = run_smart_ensemble(frame)

    if not is_image:
        wbf_boxes, wbf_labels = pos_smoother.update(wbf_boxes, wbf_labels)
        neg_boxes, neg_labels = neg_smoother.update(neg_boxes, neg_labels)

    persons, hats, vests, masks, no_hats, no_vests = [], [], [], [], [], []
    falls = 0

    chk_ppe = STATE.mode in ["unified", "ppe_all", "ppe_hat", "ppe_hat_vest", "cctv"]
    req_hat = STATE.mode in ["unified", "ppe_all", "ppe_hat", "ppe_hat_vest", "cctv"]
    req_vest = STATE.mode in ["unified", "ppe_all", "ppe_hat_vest", "cctv"]
    req_mask = STATE.mode in ["unified", "ppe_all", "cctv"]
    chk_ergo = STATE.mode in ["unified", "ergo_only"]

    for i in range(len(wbf_boxes)):
        cls_id = int(wbf_labels[i])
        coords = [int(wbf_boxes[i][0]*w), int(wbf_boxes[i][1]*h), int(wbf_boxes[i][2]*w), int(wbf_boxes[i][3]*h)]
        if ((coords[2] - coords[0]) * (coords[3] - coords[1])) < 20: continue 
        if cls_id == 0: persons.append(coords)
        elif cls_id == 1 and chk_ppe: hats.append(coords); cv2.rectangle(frame, (coords[0],coords[1]), (coords[2],coords[3]), (255,255,0), 2)
        elif cls_id == 2 and chk_ppe: masks.append(coords); cv2.rectangle(frame, (coords[0],coords[1]), (coords[2],coords[3]), (255,0,255), 2) 
        elif cls_id == 3 and chk_ppe: vests.append(coords); cv2.rectangle(frame, (coords[0],coords[1]), (coords[2],coords[3]), (0,165,255), 2)
        elif cls_id == 5: falls += 1

    for i in range(len(neg_boxes)):
        cls_id = int(neg_labels[i])
        coords = [int(neg_boxes[i][0]*w), int(neg_boxes[i][1]*h), int(neg_boxes[i][2]*w), int(neg_boxes[i][3]*h)]
        if cls_id == 2: no_hats.append(coords)
        elif cls_id == 4: no_vests.append(coords)

    if chk_ppe:
        for v in vests:
            if not any(check_ioa(v, p) for p in persons):
                vw, vh = v[2]-v[0], v[3]-v[1]
                persons.append([max(0, v[0]-15), max(0, v[1]-int(vh*0.4)), v[2]+15, v[3]+int(vh*0.8)]) 
        for hat in hats:
            if not any(check_ioa(hat, p, is_hat=True) for p in persons):
                hw, hh = hat[2]-hat[0], hat[3]-hat[1]
                persons.append([max(0, hat[0]-int(hw*0.5)), hat[1], hat[2]+int(hw*0.5), hat[3]+int(hh*6)]) 

    miss_hat, miss_vest, miss_mask, ergo_pts = 0, 0, 0, 0
    gate_denied = False

    for i, p_box in enumerate(persons):
        px1, py1, px2, py2 = p_box
        missing = []
        if chk_ppe:
            if req_hat and not any(check_ioa(b, p_box, is_hat=True) for b in hats): 
                missing.append("Hat"); miss_hat += 1
                drawn = False
                for b in no_hats:
                    if check_ioa(b, p_box, is_hat=True): cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), (0,0,255), 2); drawn = True
                if not drawn: 
                    h_y1, h_y2 = py1, py1 + int((py2 - py1) * 0.15)
                    h_x1, h_x2 = px1 + int((px2 - px1) * 0.2), px2 - int((px2 - px1) * 0.2)
                    cv2.rectangle(frame, (h_x1, h_y1), (h_x2, h_y2), (0,0,255), 1)

            if req_vest and not any(check_ioa(b, p_box) for b in vests): 
                missing.append("Vest"); miss_vest += 1
                drawn = False
                for b in no_vests:
                    if check_ioa(b, p_box): cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), (0,0,255), 2); drawn = True
                if not drawn:
                    v_y1, v_y2 = py1 + int((py2 - py1) * 0.2), py1 + int((py2 - py1) * 0.55)
                    v_x1, v_x2 = px1 + int((px2 - px1) * 0.1), px2 - int((px2 - px1) * 0.1)
                    cv2.rectangle(frame, (v_x1, v_y1), (v_x2, v_y2), (0,0,255), 1)

            if req_mask and not any(check_ioa(b, p_box) for b in masks): missing.append("Mask"); miss_mask += 1
            if len(missing) > 0: gate_denied = True

        cx, cy = (px1 + px2) // 2, (py1 + py2) // 2
        matched_id = None
        for tid, data in temporal_trackers.items():
            if math.hypot(cx - data['center'][0], cy - data['center'][1]) < 100:
                matched_id = tid; temporal_trackers[tid]['center'] = (cx, cy); break
        if not matched_id:
            matched_id = next_tracker_id; temporal_trackers[matched_id] = {'center': (cx, cy), 'bad_frames': 0}
            next_tracker_id += 1

        ergo_txt, ergo_color = "", (0, 255, 0)
        bad_pose, h_pen, w_risk, w_joint = False, 0, "Safe", ""

        if chk_ergo and (py2 - py1) > 60:
            person_crop = frame[py1:py2, px1:px2]
            pose_res = MODELS.m_pose.predict(person_crop, verbose=False)[0]
            if len(pose_res.keypoints) > 0:
                kps = pose_res.keypoints.data[0].cpu().numpy()
                for p1, p2 in SKELETON_PAIRS:
                    if kps[p1][2] > 0.5 and kps[p2][2] > 0.5: cv2.line(frame, (int(kps[p1][0]+px1), int(kps[p1][1]+py1)), (int(kps[p2][0]+px1), int(kps[p2][1]+py1)), (0, 200, 255), 2)
                for k in kps:
                    if k[2] > 0.5: cv2.circle(frame, (int(k[0]+px1), int(k[1]+py1)), 4, (0, 255, 255), -1)

                angles = calculate_ergonomics(kps)
                
                # --- REBA MEDICAL THRESHOLDS INtegration ---
                # 1. Trunk Flexion
                if "trunk" in angles:
                    if angles["trunk"] > 20 and angles["trunk"] <= 45: 
                        bad_pose, h_pen, w_risk, w_joint = True, 5, "Moderate", "Trunk"
                    elif angles["trunk"] > 45: 
                        bad_pose, h_pen, w_risk, w_joint = True, STATE.w_ergo, "High", "Trunk"
                
                # 2. Neck Flexion
                if "neck" in angles and angles["neck"] > 20 and w_risk != "High":
                    bad_pose, h_pen, w_risk, w_joint = True, 10, "High", "Neck"
                    
                # 3. Shoulder Elevation
                sh_angle = max(angles.get("shoulder_l", 0), angles.get("shoulder_r", 0))
                if sh_angle > 90 and w_risk != "High":
                    bad_pose, h_pen, w_risk, w_joint = True, STATE.w_ergo, "High", "Shoulder"
                    
                # 4. Knee Flexion (Squatting)
                knee_angle = min(angles.get("knee_l", 180), angles.get("knee_r", 180))
                if knee_angle < 60 and w_risk != "High":
                    bad_pose, h_pen, w_risk, w_joint = True, 10, "High", "Knees"

        if bad_pose: temporal_trackers[matched_id]['bad_frames'] += 1
        else: temporal_trackers[matched_id]['bad_frames'] = max(0, temporal_trackers[matched_id]['bad_frames'] - 1)

        if temporal_trackers[matched_id]['bad_frames'] >= 5:
            ergo_pts += h_pen
            ergo_txt = f" | {w_joint}: {w_risk}"
            ergo_color = (0, 165, 255) if w_risk == "Moderate" else (0, 0, 255)

        # =====================================================================
        if STATE.mode != "cctv" and (len(missing) > 0 or ergo_pts > 0):
            current_time = time.time()
            
            if matched_id not in STATE.worker_violations:
                STATE.worker_violations[matched_id] = 0
                STATE.last_logged_time[matched_id] = 0
                
            if current_time - STATE.last_logged_time[matched_id] > 5.0:
                STATE.last_logged_time[matched_id] = current_time
                STATE.worker_violations[matched_id] += 1
                
                # Use a single datetime object to sync the CSV timestamp and the Image filename perfectly
                now_dt = datetime.now()
                csv_time = now_dt.strftime('%H:%M:%S')
                file_time = now_dt.strftime('%H%M%S')
                
                violation_type = f"Missing: {','.join(missing)}" if missing else f"Ergo: {w_risk}"
                
                with open(STATE.csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([csv_time, f"ID_{matched_id}", violation_type, STATE.telemetry['score']])
                
                try:
                    pad = 20
                    crop_y1, crop_y2 = max(0, py1-pad), min(h, py2+pad)
                    crop_x1, crop_x2 = max(0, px1-pad), min(w, px2+pad)
                    incident_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    snap_name = os.path.join(INCIDENTS_FOLDER, f"ID{matched_id}_{file_time}.jpg")
                    cv2.imwrite(snap_name, incident_crop)
                except Exception as e:
                    pass 
                
                if STATE.worker_violations[matched_id] == 3:
                    msg = f"Worker ID {matched_id} has accumulated 3 safety violations in this shift."
                    fire_rpa_alert("WARNING", msg, {"worker_id": matched_id, "last_violation": violation_type})
        # =====================================================================

        if STATE.mode == "cctv": pass 
        else:
            box_color = (0, 0, 255) if len(missing) > 0 else ergo_color
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, f"ID{matched_id}: {'SAFE' if not missing else 'No '+','.join(missing)}{ergo_txt}", (px1, max(0, py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    if STATE.mode == "cctv" and len(persons) > 0:
        if gate_denied: cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 20); cv2.putText(frame, "ACCESS DENIED", (w//2-200, h//2), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,255), 4)
        else: cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 20); cv2.putText(frame, "ACCESS GRANTED", (w//2-200, h//2), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,255,0), 4)

    score = 100
    if len(persons) > 0:
        base = 0
        if req_hat: base += STATE.w_hat
        if req_vest: base += STATE.w_vest
        if req_mask: base += STATE.w_mask 
        if chk_ergo: base += STATE.w_ergo
        if base == 0: base = 1 
        max_p = len(persons) * base
        act_p = (miss_hat * STATE.w_hat) + (miss_vest * STATE.w_vest) + (miss_mask * STATE.w_mask) + ergo_pts
        score = max(0, int(100 - ((act_p / max_p) * 100)))

    if falls > 0: 
        score = 0
        fire_rpa_alert("CRITICAL", "FALL DETECTED! Immediate assistance required.", {"falls_detected": falls})

    STATE.telemetry = {"score": score, "workers": len(persons), "hats": miss_hat, "vests": miss_vest, "masks": miss_mask, "ergo": ergo_pts, "falls": falls}
    return frame, next_tracker_id

def process_video_background(filepath):
    STATE.bg_status = "processing"
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened(): STATE.bg_status = "error"; return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    STATE.bg_total_frames = total_frames if total_frames > 0 else 1
    
    ret, test_frame = cap.read()
    if not ret: return
    h, w = test_frame.shape[:2]
    if h > 720 or w > 1080: scale = min(1080/w, 720/h); w, h = int(w*scale), int(h*scale)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

    dash_w = max(340, int(w * 0.35))
    final_w = w + dash_w
    out_norm = os.path.join(EXPORT_FOLDER, f"BG_Normal_{datetime.now().strftime('%H%M%S')}.mp4")
    out_slow = os.path.join(EXPORT_FOLDER, f"BG_SlowMo_{datetime.now().strftime('%H%M%S')}.mp4")
    
    v_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_norm = v_fps if v_fps > 0 else 25
    fps_slow = max(1, int(fps_norm / 3))

    w_norm = cv2.VideoWriter(out_norm, cv2.VideoWriter_fourcc(*'mp4v'), fps_norm, (final_w, h))
    w_slow = cv2.VideoWriter(out_slow, cv2.VideoWriter_fourcc(*'mp4v'), fps_slow, (final_w, h))

    pos_sm, neg_sm = BoundingBoxSmoother(), BoundingBoxSmoother()
    trackers, next_id, current_frame, start_time = {}, 1, 0, time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        current_frame += 1
        STATE.bg_current_frame = current_frame
        if frame.shape[0] != h or frame.shape[1] != w: frame = cv2.resize(frame, (w, h))
        
        frame, next_id = process_single_frame(frame, pos_sm, neg_sm, trackers, next_id, is_image=False)
        combined = append_dashboard(frame, STATE.telemetry, STATE.mode)
        
        w_norm.write(combined)
        w_slow.write(combined)

        if current_frame % 10 == 0: 
            elapsed = time.time() - start_time
            fps_proc = current_frame / elapsed
            m, s = divmod(int((total_frames - current_frame) / fps_proc), 60)
            STATE.bg_eta = f"{m:02d}:{s:02d}"

    cap.release(); w_norm.release(); w_slow.release()
    STATE.bg_status = "complete"
    STATE.bg_message = f"Saved Normal & Slow-Mo exports to folder."

def generate_frames():
    my_stream_id = time.time()
    STATE.stream_id = my_stream_id
    STATE.is_recording = True
    
    pos_smoother, neg_smoother = BoundingBoxSmoother(), BoundingBoxSmoother()
    temporal_trackers, next_tracker_id = {}, 1

    is_image = (STATE.input_type == "image")
    cap, frame_input = None, None

    if is_image:
        if STATE.file_path and os.path.exists(STATE.file_path): frame_input = cv2.imread(STATE.file_path)
        if frame_input is None:
            frame_input = np.zeros((720, 1080, 3), dtype=np.uint8)
            cv2.putText(frame_input, "AWAITING UPLOAD...", (250, 360), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 200), 2)
    elif STATE.input_type == "video":
        if STATE.file_path and os.path.exists(STATE.file_path): cap = cv2.VideoCapture(STATE.file_path)
    else: cap = cv2.VideoCapture(0)

    while STATE.is_recording and STATE.stream_id == my_stream_id:
        if is_image:
            frame = frame_input.copy()
            time.sleep(0.5) 
        else:
            if cap is None: break
            success, frame = cap.read()
            if not success: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        h, w = frame.shape[:2]
        if h > 720 or w > 1080: scale = min(1080/w, 720/h); frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        if STATE.file_path or STATE.input_type == "webcam":
            frame, next_tracker_id = process_single_frame(frame, pos_smoother, neg_smoother, temporal_trackers, next_tracker_id, is_image)
            combined_frame = append_dashboard(frame, STATE.telemetry, STATE.mode)
            STATE.last_saved_frame = combined_frame.copy() 

            if not is_image and STATE.writer_normal is None:
                v_h, v_w = combined_frame.shape[:2]
                out_n = os.path.join(EXPORT_FOLDER, f"Stream_Normal_{datetime.now().strftime('%H%M%S')}.mp4")
                out_s = os.path.join(EXPORT_FOLDER, f"Stream_SlowMo_{datetime.now().strftime('%H%M%S')}.mp4")
                v_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 25
                v_fps = v_fps if v_fps > 0 else 25
                STATE.writer_normal = cv2.VideoWriter(out_n, cv2.VideoWriter_fourcc(*'mp4v'), v_fps, (v_w, v_h))
                STATE.writer_slow = cv2.VideoWriter(out_s, cv2.VideoWriter_fourcc(*'mp4v'), max(1, int(v_fps/3)), (v_w, v_h))

            if STATE.writer_normal is not None:
                STATE.writer_normal.write(combined_frame)
                STATE.writer_slow.write(combined_frame)

            ret, buffer = cv2.imencode('.jpg', combined_frame)
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# =====================================================================
# --- PHASE 2: AUTOMATED PDF REPORT GENERATOR (WITH FORENSIC IMAGES) ---
# =====================================================================
def generate_pdf_report():
    if not os.path.exists(STATE.csv_path):
        return None

    try:
        df = pd.read_csv(STATE.csv_path)
        if df.empty:
            return None

        avg_score = df["System_Score"].mean()
        total_incidents = len(df)
        violation_counts = df["Violation_Type"].value_counts()

        charts_dir = os.path.join(REPORTS_FOLDER, 'temp_charts')
        os.makedirs(charts_dir, exist_ok=True)

        # 1. Generate Pie Chart
        plt.figure(figsize=(6, 6))
        if not violation_counts.empty:
            violation_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        plt.title('Violation Distribution', color='black')
        plt.ylabel('')
        pie_path = os.path.join(charts_dir, f'pie_{STATE.shift_id}.png')
        plt.savefig(pie_path, bbox_inches='tight')
        plt.close()

        # 2. Generate Line Chart
        plt.figure(figsize=(8, 4))
        plt.plot(df.index, df['System_Score'], marker='o', color='#da3633', linestyle='-')
        plt.title('Site Safety Score Over Time (Incident Sequence)')
        plt.xlabel('Incident #')
        plt.ylabel('Score (0-100)')
        plt.ylim(0, 105)
        plt.grid(True, linestyle='--', alpha=0.6)
        line_path = os.path.join(charts_dir, f'line_{STATE.shift_id}.png')
        plt.savefig(line_path, bbox_inches='tight')
        plt.close()

        # 3. Build the PDF Document - PAGE 1 (Summary)
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Arial", size=20, style='B')
        pdf.cell(200, 15, txt="Executive Daily Safety Digest", ln=True, align='C')
        
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 8, txt=f"Shift ID / Date: {STATE.shift_id}", ln=True, align='C')
        pdf.cell(200, 8, txt=f"Total Incidents Logged: {total_incidents}", ln=True, align='C')
        
        if avg_score > 80:
            pdf.set_text_color(35, 134, 54) 
        elif avg_score > 50:
            pdf.set_text_color(210, 153, 34) 
        else:
            pdf.set_text_color(218, 54, 51) 
            
        pdf.set_font("Arial", size=14, style='B')
        pdf.cell(200, 10, txt=f"Average Shift Compliance Score: {avg_score:.1f} / 100", ln=True, align='C')
        pdf.set_text_color(0, 0, 0) 
        
        if os.path.exists(pie_path):
            pdf.image(pie_path, x=10, y=70, w=90)
        if os.path.exists(line_path):
            pdf.image(line_path, x=105, y=75, w=95)

        # 4. Build the PDF Document - PAGE 2+ (Incident Evidence Log)
        pdf.add_page()
        pdf.set_font("Arial", size=16, style='B')
        pdf.cell(200, 10, txt="Detailed Incident Log & Forensic Evidence", ln=True, align='L')
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(200, 6, txt="The following events triggered RPA alerts and audit logging during this shift.", ln=True, align='L')
        pdf.ln(5)
        pdf.set_text_color(0, 0, 0)

        # Loop through CSV to attach the exact images and details
        for index, row in df.iterrows():
            if pdf.get_y() > 230: # Prevent writing off the bottom of the page
                pdf.add_page()
            
            y_before = pdf.get_y()
            
            # Reconstruct the exact image filename from the CSV timestamp
            t_csv = str(row['Timestamp'])
            t_file = t_csv.replace(":", "")
            w_id = str(row['Worker_ID']).replace("ID_", "ID")
            img_path = os.path.join(INCIDENTS_FOLDER, f"{w_id}_{t_file}.jpg")
            
            # Embed Image
            if os.path.exists(img_path):
                try:
                    pdf.image(img_path, x=10, y=y_before, h=40)
                except:
                    pass 
            else:
                pdf.rect(x=10, y=y_before, w=30, h=40)
                pdf.text(x=12, y=y_before+20, txt="No Image")

            # Embed Text Details
            pdf.set_xy(60, y_before + 5)
            pdf.set_font("Arial", size=12, style='B')
            
            if "Ergo" in row['Violation_Type']:
                pdf.set_text_color(210, 153, 34) 
            else:
                pdf.set_text_color(218, 54, 51) 
                
            pdf.cell(100, 8, txt=f"Incident #{index+1} - {row['Violation_Type']}", ln=True)
            pdf.set_text_color(0, 0, 0)
            
            pdf.set_xy(60, y_before + 15)
            pdf.set_font("Arial", size=10)
            pdf.cell(100, 6, txt=f"Time Detected: {t_csv} | Target: {row['Worker_ID']}", ln=True)
            
            pdf.set_xy(60, y_before + 21)
            pdf.cell(100, 6, txt=f"Site Safety Score Impacted: {row['System_Score']} / 100", ln=True)
            
            pdf.set_xy(60, y_before + 27)
            pdf.set_font("Arial", size=9, style='I')
            pdf.set_text_color(100, 100, 100)
            pdf.cell(100, 6, txt="Action: Logged to Audit Trail & RPA Webhook Evaluated.", ln=True)
            pdf.set_text_color(0, 0, 0)
            
            # Draw separating line
            pdf.set_y(y_before + 45)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

        pdf_path = os.path.join(REPORTS_FOLDER, f"Executive_Report_{STATE.shift_id}.pdf")
        pdf.output(pdf_path)

        fire_rpa_alert("ADMINISTRATIVE", f"Shift closed. Avg Score: {avg_score:.1f}. PDF generated.", {"report": pdf_path})
        
        return pdf_path

    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None
# =====================================================================

@app.route('/')
@app.route('/<mode>')
def index(mode="unified"):
    titles = {"unified": "Unified Engine (PPE + Ergo)", "ppe_all": "PPE Compliance: Strict", "ppe_hat_vest": "PPE Compliance: Hat & Vest", "ergo_only": "Ergonomics Analysis", "cctv": "CCTV Gatekeeper Mode"}
    if mode in titles: STATE.mode = mode
    return render_template('index.html', mode=STATE.mode, title=titles.get(STATE.mode, "Command Center"), input_type=STATE.input_type)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_input', methods=['POST'])
def set_input():
    STATE.input_type = request.json.get("type", "webcam")
    if STATE.input_type == "webcam": STATE.file_path = 0
    return jsonify({"status": "success"})

@app.route('/switch_models', methods=['POST'])
def switch_models():
    scale = request.json.get("scale", "medium")
    MODELS.load_models(scale)
    return jsonify({"status": "success", "message": f"Successfully loaded {scale.upper()} architecture."})

@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    STATE.w_hat = int(data.get("w_hat", 20))
    STATE.w_vest = int(data.get("w_vest", 15))
    STATE.w_mask = int(data.get("w_mask", 5))
    STATE.w_ergo = int(data.get("w_ergo", 15))
    return jsonify({"status": "success"})

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    upload_type = request.form.get("type", "image")
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        if upload_type == "bg_video":
            STATE.input_type = "bg_video"
            threading.Thread(target=process_video_background, args=(filepath,), daemon=True).start()
            return jsonify({"status": "background_started"})
        else:
            STATE.input_type = upload_type
            STATE.file_path = filepath
            return jsonify({"status": "success"})
    return jsonify({"error": "File was not received."})

@app.route('/stop_and_save', methods=['POST'])
def stop_and_save():
    STATE.is_recording = False
    msg = "Processing stopped."
    
    pdf_path = generate_pdf_report()
    pdf_msg = f" | PDF Report Generated!" if pdf_path else ""
    
    if STATE.input_type == "image" and STATE.last_saved_frame is not None:
        out_path = os.path.join(EXPORT_FOLDER, f"Export_Img_HUD_{datetime.now().strftime('%H%M%S')}.jpg")
        cv2.imwrite(out_path, STATE.last_saved_frame)
        msg = f"SUCCESS: Burned-In Image exported to exports folder" + pdf_msg
    elif STATE.writer_normal is not None:
        STATE.writer_normal.release()
        STATE.writer_slow.release()
        STATE.writer_normal = None
        STATE.writer_slow = None
        msg = "SUCCESS: Dual Video Streams (Normal & Slow-Mo) safely exported." + pdf_msg
        
    return jsonify({"message": msg})

@app.route('/stats')
def get_stats(): return jsonify(STATE.telemetry)

@app.route('/bg_status')
def get_bg_status(): return jsonify({"status": STATE.bg_status, "current": STATE.bg_current_frame, "total": STATE.bg_total_frames, "eta": STATE.bg_eta, "message": STATE.bg_message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)