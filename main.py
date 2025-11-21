from typing import Dict, Any

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException

# -------- 1. Load artifacts --------

scale_cols = [
    "Nhiet_do_Trung_binh (Celsius)",
    "Nhiet_do_Trung_binh_Cao_nhat (Celsius)",
    "Nhiet_do_Trung_binh_Thap_nhat (Celsius)",
    "Do am Tuong doi (%)",
    "Tong Luong mua (mm)",
    "Temp_Range",
    "Dientich (nghin ha)",
]

SCALER_PATH = "./scaler.pkl"
ENCODERS_PATH = "./encoders.pkl"
MODEL_PATH = "./xgb_model.pkl"

scaler = joblib.load(SCALER_PATH)
encoders: Dict[str, Any] = joblib.load(ENCODERS_PATH)
model = joblib.load(MODEL_PATH)

mapping_vung: Dict[str, int] = encoders["Vung"]
mapping_cay: Dict[str, int] = encoders["Cay"]
crop_col: str = encoders.get("crop_col", "Cay")  # nếu bạn lưu tên cột khác thì lấy từ đây


# -------- 2. Core logic --------

def prepare_features_from_dict(row_dict: Dict[str, Any]) -> np.ndarray:
    """
    row_dict: dict input với các key giống lúc train:
      - các cột số: scale_cols
      - 'Vung'
      - cột cây trồng: crop_col (vd: 'Cay')
    """
    # 1) Kiểm tra cột số
    missing = [c for c in scale_cols if c not in row_dict]
    if missing:
        raise KeyError(f"Thiếu các cột số cần chuẩn hóa: {missing}")

    # 2) Kiểm tra cột 'Vung' và cây trồng
    if "Vung" not in row_dict or crop_col not in row_dict:
        raise KeyError(f"Phải cung cấp cả 'Vung' và '{crop_col}' trong input.")

    # 3) Chuẩn bị numeric theo đúng thứ tự
    try:
        numeric_values = [float(row_dict[c]) for c in scale_cols]
    except ValueError as e:
        raise ValueError(f"Lỗi chuyển kiểu dữ liệu số: {e}")

    arr_scale = np.array([numeric_values], dtype=float)  # shape (1, len(scale_cols))

    # 4) Chuẩn hóa
    arr_scale_scaled = scaler.transform(arr_scale)

    # 5) Encode 'Vung'
    vung_val = row_dict["Vung"]
    if vung_val not in mapping_vung:
        raise ValueError(f"Giá trị Vung chưa có trong encoder: {vung_val}")
    vung_code = mapping_vung[vung_val]

    # 6) Encode cây trồng
    cay_val = row_dict[crop_col]
    if cay_val not in mapping_cay:
        raise ValueError(f"Giá trị '{crop_col}' chưa có trong encoder: {cay_val}")
    cay_code = mapping_cay[cay_val]

    # 7) Gộp thành feature vector
    final = np.concatenate([arr_scale_scaled.flatten(), [vung_code, cay_code]])
    return final.reshape(1, -1)


def predict_row(row_dict: Dict[str, Any]) -> float:
    feats = prepare_features_from_dict(row_dict)
    pred = model.predict(feats)
    return float(pred[0])


# -------- 3. FastAPI app --------

app = FastAPI(
    title="Agriculture Yield Prediction API",
    description="Dự báo sản lượng (nghìn tấn) từ dữ liệu khí hậu & cây trồng.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Agriculture XGBoost prediction API is running."}


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    Ví dụ body JSON:

    {
      "Nhiet_do_Trung_binh (Celsius)": 27.0,
      "Nhiet_do_Trung_binh_Cao_nhat (Celsius)": 30.0,
      "Nhiet_do_Trung_binh_Thap_nhat (Celsius)": 23.0,
      "Do am Tuong doi (%)": 78.0,
      "Tong Luong mua (mm)": 150.0,
      "Temp_Range": 7.0,
      "Dientich (nghin ha)": 2.5,
      "Vung": "An Giang",
      "LoaiCayTrong": "Khoai"   // hoặc key đúng với crop_col trong encoders.pkl
    }
    """
    try:
        y_hat = predict_row(payload)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return {
        "input": payload,
        "prediction": y_hat,
        "unit": "Sanluong nghin tan",
    }
