## Hướng dẫn

```shell
# 1) jump folder project
cd team6-data-mining-agri-fastapi

# 2) Tạo virtual env (.venv)
python -m venv .venv

# 3) active (Windows (cmd)):
.\.venv\Scripts\activate.bat

# 4) install packages
pip install -r requirements.txt
```

```shell
uvicorn main:app --reload --port 8000
```

```text
1. Truy cập swagger
http://127.0.0.1:8000/docs

2. Chọn api predict

3. Try it out

4. Nhập reqquest bên dưới

5. Bấm Execute
```
Sample predict

```json
{
  "Nhiet_do_Trung_binh (Celsius)": 27.0,
  "Nhiet_do_Trung_binh_Cao_nhat (Celsius)": 30.0,
  "Nhiet_do_Trung_binh_Thap_nhat (Celsius)": 23.0,
  "Do am Tuong doi (%)": 78.0,
  "Tong Luong mua (mm)": 150.0,
  "Temp_Range": 7.0,
  "Dientich (nghin ha)": 2.5,
  "Vung": "An Giang",
  "LoaiCayTrong": "Khoai"
}
``