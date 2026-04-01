import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.interpolate import interp1d
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import warnings
warnings.filterwarnings('ignore')

CURRENT_DATE = datetime.now()

st.set_page_config(page_title="중고 애플 기기 가격 예측", page_icon="📱", layout="wide")

st.markdown("""
<style>
    /* ===== A. 전체 글씨 크기 ===== */
    .stMainBlockContainer, .block-container { font-size: 1.1rem; }
    section[data-testid="stSidebar"] { font-size: 1.05rem; }

    /* ===== B. 메트릭 카드 스타일 ===== */
    [data-testid="stMetricValue"] {
        font-size: 2.0rem !important;
        font-weight: 700 !important;
        color: #0D47A1 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #1565C0 !important;
    }
    /* 메트릭 컨테이너 카드화 */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(13, 71, 161, 0.1);
        border-left: 4px solid #1565C0;
    }
    /* ===== C. 섹션 제목 강화 ===== */
    [data-testid="stSubheader"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #0D47A1 !important;
        border-bottom: 3px solid #1565C0;
        padding-bottom: 8px;
        margin-top: 24px !important;
        margin-bottom: 4px !important;
    }

    /* ===== D. 캡션 (모델 설명) ===== */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 0.92rem !important;
        color: #5C6BC0 !important;
        font-style: italic;
        margin-bottom: 16px !important;
    }

    /* ===== E. info 배너 스타일 ===== */
    .stAlert {
        border-radius: 10px !important;
        border-left: 5px solid #1565C0 !important;
        background-color: #E8EAF6 !important;
        margin-bottom: 20px !important;
    }
    .stAlert p {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #1A237E !important;
        line-height: 1.7 !important;
    }

    /* ===== F. 테이블 스타일 ===== */
    .stTable {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }
    .stTable thead th {
        background-color: #1565C0 !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 10px 12px !important;
    }
    .stTable tbody td {
        font-size: 0.95rem !important;
        padding: 8px 12px !important;
        border-bottom: 1px solid #E3F2FD !important;
    }
    .stTable tbody tr:hover {
        background-color: #E3F2FD !important;
    }

    /* ===== G. 구분선 강화 ===== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, #BBDEFB, #1565C0, #BBDEFB);
        margin: 28px 0;
    }

    /* ===== H. 차트 영역 여백 ===== */
    [data-testid="stVegaLiteChart"],
    .stPlotlyChart {
        background: #FAFAFA;
        border-radius: 10px;
        padding: 8px;
        margin: 12px 0 20px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* ===== I. 전체 여백/호흡 ===== */
    .stMainBlockContainer .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* ===== J. 사이드바 스타일 ===== */
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
        font-weight: 700;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label {
        font-weight: 500 !important;
    }

    /* ===== K. warning 스타일 ===== */
    [data-testid="stAlert"][data-baseweb*="warning"] {
        border-left: 5px solid #F57F17 !important;
    }
</style>
""", unsafe_allow_html=True)

# 정상 조합 테이블 (공식 스펙 기반)
VALID_COMBOS = {
    'pro': {
        '1세대': {'sizes': [9.7, 12.9], 'storages': [32, 128, 256]},
        '2세대': {'sizes': [10.5, 12.9], 'storages': [64, 256, 512]},
        '3세대': {'sizes': [11.0, 12.9], 'storages': [64, 256, 512, 1024]},
        '4세대': {'sizes': [11.0, 12.9], 'storages': [128, 256, 512, 1024]},
        'm1': {'sizes': [11.0, 12.9], 'storages': [128, 256, 512, 1024, 2048]},
        'm2': {'sizes': [11.0, 12.9], 'storages': [128, 256, 512, 1024, 2048]},
        'm4': {'sizes': [11.0, 13.0], 'storages': [256, 512, 1024, 2048]},
        'm5': {'sizes': [11.0, 13.0], 'storages': [256, 512, 1024, 2048]},
    },
    'air': {
        '1세대': {'sizes': [9.7], 'storages': [16, 32, 64, 128]},
        '2세대': {'sizes': [9.7], 'storages': [16, 32, 64, 128]},
        '3세대': {'sizes': [10.5], 'storages': [64, 256]},
        '4세대': {'sizes': [10.9], 'storages': [64, 256]},
        'm1': {'sizes': [10.9], 'storages': [64, 256]},
        'm2': {'sizes': [11.0, 13.0], 'storages': [128, 256, 512, 1024]},
        'm3': {'sizes': [11.0, 13.0], 'storages': [128, 256, 512, 1024]},
    },
    'basic': {
        '5세대': {'sizes': [9.7], 'storages': [32, 128]},
        '6세대': {'sizes': [9.7], 'storages': [32, 128]},
        '7세대': {'sizes': [10.2], 'storages': [32, 128]},
        '8세대': {'sizes': [10.2], 'storages': [32, 128]},
        '9세대': {'sizes': [10.2], 'storages': [64, 256]},
        '10세대': {'sizes': [10.9], 'storages': [64, 256]},
        '11세대': {'sizes': [11.0], 'storages': [128, 256, 512]},
    },
    'mini': {
        '2세대': {'sizes': [7.9], 'storages': [16, 32, 64, 128]},
        '4세대': {'sizes': [7.9], 'storages': [16, 32, 64, 128]},
        '5세대': {'sizes': [7.9], 'storages': [64, 256]},
        '6세대': {'sizes': [8.3], 'storages': [64, 256]},
        '7세대': {'sizes': [8.3], 'storages': [128, 256, 512]},
    },
}

# 출시가 테이블 (공식 가격)
LAUNCH_PRICES = {
    # PRO
    ('pro','1세대',9.7,32,'wifi'): 760000, ('pro','1세대',9.7,128,'wifi'): 960000,
    ('pro','1세대',9.7,256,'wifi'): 1160000, ('pro','1세대',9.7,32,'cellular'): 930000,
    ('pro','1세대',9.7,128,'cellular'): 1130000, ('pro','1세대',9.7,256,'cellular'): 1330000,
    ('pro','1세대',12.9,32,'wifi'): 1050000, ('pro','1세대',12.9,128,'wifi'): 1200000,
    ('pro','1세대',12.9,256,'wifi'): 1400000, ('pro','1세대',12.9,32,'cellular'): 1220000,
    ('pro','1세대',12.9,128,'cellular'): 1350000, ('pro','1세대',12.9,256,'cellular'): 1550000,
    ('pro','2세대',10.5,64,'wifi'): 799000, ('pro','2세대',10.5,256,'wifi'): 919000,
    ('pro','2세대',10.5,512,'wifi'): 1159000, ('pro','2세대',10.5,64,'cellular'): 969000,
    ('pro','2세대',10.5,256,'cellular'): 1089000, ('pro','2세대',10.5,512,'cellular'): 1329000,
    ('pro','2세대',12.9,64,'wifi'): 999000, ('pro','2세대',12.9,256,'wifi'): 1120000,
    ('pro','2세대',12.9,512,'wifi'): 1360000, ('pro','2세대',12.9,64,'cellular'): 1170000,
    ('pro','2세대',12.9,256,'cellular'): 1290000, ('pro','2세대',12.9,512,'cellular'): 1530000,
    ('pro','3세대',11.0,64,'wifi'): 999000, ('pro','3세대',11.0,256,'wifi'): 1129000,
    ('pro','3세대',11.0,512,'wifi'): 1389000, ('pro','3세대',11.0,1024,'wifi'): 1909000,
    ('pro','3세대',11.0,64,'cellular'): 1199000, ('pro','3세대',11.0,256,'cellular'): 1329000,
    ('pro','3세대',11.0,512,'cellular'): 1589000, ('pro','3세대',11.0,1024,'cellular'): 2109000,
    ('pro','3세대',12.9,64,'wifi'): 1269000, ('pro','3세대',12.9,256,'wifi'): 1469000,
    ('pro','3세대',12.9,512,'wifi'): 1739000, ('pro','3세대',12.9,1024,'wifi'): 2279000,
    ('pro','3세대',12.9,64,'cellular'): 1469000, ('pro','3세대',12.9,256,'cellular'): 1669000,
    ('pro','3세대',12.9,512,'cellular'): 1939000, ('pro','3세대',12.9,1024,'cellular'): 2479000,
    ('pro','4세대',11.0,128,'wifi'): 1029000, ('pro','4세대',11.0,256,'wifi'): 1159000,
    ('pro','4세대',11.0,512,'wifi'): 1419000, ('pro','4세대',11.0,1024,'wifi'): 1949000,
    ('pro','4세대',11.0,128,'cellular'): 1229000, ('pro','4세대',11.0,256,'cellular'): 1359000,
    ('pro','4세대',11.0,512,'cellular'): 1619000, ('pro','4세대',11.0,1024,'cellular'): 2149000,
    ('pro','4세대',12.9,128,'wifi'): 1299000, ('pro','4세대',12.9,256,'wifi'): 1429000,
    ('pro','4세대',12.9,512,'wifi'): 1689000, ('pro','4세대',12.9,1024,'wifi'): 1949000,
    ('pro','4세대',12.9,128,'cellular'): 1499000, ('pro','4세대',12.9,256,'cellular'): 1629000,
    ('pro','4세대',12.9,512,'cellular'): 1889000, ('pro','4세대',12.9,1024,'cellular'): 2149000,
    ('pro','m1',11.0,128,'wifi'): 999000, ('pro','m1',11.0,256,'wifi'): 1129000,
    ('pro','m1',11.0,512,'wifi'): 1389000, ('pro','m1',11.0,1024,'wifi'): 1909000,
    ('pro','m1',11.0,2048,'wifi'): 2429000,
    ('pro','m1',11.0,128,'cellular'): 1199000, ('pro','m1',11.0,256,'cellular'): 1329000,
    ('pro','m1',11.0,512,'cellular'): 1589000, ('pro','m1',11.0,1024,'cellular'): 2109000,
    ('pro','m1',11.0,2048,'cellular'): 2629000,
    ('pro','m1',12.9,128,'wifi'): 1379000, ('pro','m1',12.9,256,'wifi'): 1509000,
    ('pro','m1',12.9,512,'wifi'): 1769000, ('pro','m1',12.9,1024,'wifi'): 2289000,
    ('pro','m1',12.9,2048,'wifi'): 2809000,
    ('pro','m1',12.9,128,'cellular'): 1579000, ('pro','m1',12.9,256,'cellular'): 1709000,
    ('pro','m1',12.9,512,'cellular'): 1969000, ('pro','m1',12.9,1024,'cellular'): 2489000,
    ('pro','m1',12.9,2048,'cellular'): 3009000,
    ('pro','m2',11.0,128,'wifi'): 1249000, ('pro','m2',11.0,256,'wifi'): 1399000,
    ('pro','m2',11.0,512,'wifi'): 1699000, ('pro','m2',11.0,1024,'wifi'): 2299000,
    ('pro','m2',11.0,2048,'wifi'): 2899000,
    ('pro','m2',11.0,128,'cellular'): 1489000, ('pro','m2',11.0,256,'cellular'): 1639000,
    ('pro','m2',11.0,512,'cellular'): 1939000, ('pro','m2',11.0,1024,'cellular'): 2539000,
    ('pro','m2',11.0,2048,'cellular'): 3139000,
    ('pro','m2',12.9,128,'wifi'): 1729000, ('pro','m2',12.9,256,'wifi'): 1879000,
    ('pro','m2',12.9,512,'wifi'): 2179000, ('pro','m2',12.9,1024,'wifi'): 2779000,
    ('pro','m2',12.9,2048,'wifi'): 3379000,
    ('pro','m2',12.9,128,'cellular'): 1969000, ('pro','m2',12.9,256,'cellular'): 2119000,
    ('pro','m2',12.9,512,'cellular'): 2419000, ('pro','m2',12.9,1024,'cellular'): 3019000,
    ('pro','m2',12.9,2048,'cellular'): 3619000,
    ('pro','m4',11.0,256,'wifi'): 1499000, ('pro','m4',11.0,512,'wifi'): 1899000,
    ('pro','m4',11.0,1024,'wifi'): 2499000, ('pro','m4',11.0,2048,'wifi'): 3099000,
    ('pro','m4',11.0,256,'cellular'): 1799000, ('pro','m4',11.0,512,'cellular'): 2199000,
    ('pro','m4',11.0,1024,'cellular'): 2799000, ('pro','m4',11.0,2048,'cellular'): 3399000,
    ('pro','m4',13.0,256,'wifi'): 1999000, ('pro','m4',13.0,512,'wifi'): 2399000,
    ('pro','m4',13.0,1024,'wifi'): 2999000, ('pro','m4',13.0,2048,'wifi'): 3599000,
    ('pro','m4',13.0,256,'cellular'): 2299000, ('pro','m4',13.0,512,'cellular'): 2699000,
    ('pro','m4',13.0,1024,'cellular'): 3299000, ('pro','m4',13.0,2048,'cellular'): 3899000,
    ('pro','m5',11.0,256,'wifi'): 1599000, ('pro','m5',11.0,512,'wifi'): 1899000,
    ('pro','m5',11.0,1024,'wifi'): 2499000, ('pro','m5',11.0,2048,'wifi'): 3099000,
    ('pro','m5',11.0,256,'cellular'): 1899000, ('pro','m5',11.0,512,'cellular'): 2199000,
    ('pro','m5',11.0,1024,'cellular'): 2799000, ('pro','m5',11.0,2048,'cellular'): 3399000,
    ('pro','m5',13.0,256,'wifi'): 2099000, ('pro','m5',13.0,512,'wifi'): 2399000,
    ('pro','m5',13.0,1024,'wifi'): 2999000, ('pro','m5',13.0,2048,'wifi'): 3599000,
    ('pro','m5',13.0,256,'cellular'): 2399000, ('pro','m5',13.0,512,'cellular'): 2699000,
    ('pro','m5',13.0,1024,'cellular'): 3299000, ('pro','m5',13.0,2048,'cellular'): 3899000,
    # AIR
    ('air','1세대',9.7,16,'wifi'): 620000, ('air','1세대',9.7,32,'wifi'): 740000,
    ('air','1세대',9.7,64,'wifi'): 860000, ('air','1세대',9.7,128,'wifi'): 980000,
    ('air','1세대',9.7,16,'cellular'): 689000, ('air','1세대',9.7,32,'cellular'): 798000,
    ('air','2세대',9.7,16,'wifi'): 570000, ('air','2세대',9.7,32,'wifi'): 732600,
    ('air','2세대',9.7,64,'wifi'): 720000, ('air','2세대',9.7,128,'wifi'): 720000,
    ('air','2세대',9.7,16,'cellular'): 520000, ('air','2세대',9.7,32,'cellular'): 680000,
    ('air','2세대',9.7,64,'cellular'): 880000, ('air','2세대',9.7,128,'cellular'): 974600,
    ('air','3세대',10.5,64,'wifi'): 629000, ('air','3세대',10.5,256,'wifi'): 829000,
    ('air','3세대',10.5,64,'cellular'): 799000, ('air','3세대',10.5,256,'cellular'): 999000,
    ('air','4세대',10.9,64,'wifi'): 779000, ('air','4세대',10.9,256,'wifi'): 979000,
    ('air','4세대',10.9,64,'cellular'): 949000, ('air','4세대',10.9,256,'cellular'): 1149000,
    ('air','m1',10.9,64,'wifi'): 779000, ('air','m1',10.9,256,'wifi'): 979000,
    ('air','m1',10.9,64,'cellular'): 979000, ('air','m1',10.9,256,'cellular'): 1179000,
    ('air','m2',11.0,128,'wifi'): 899000, ('air','m2',11.0,256,'wifi'): 1049000,
    ('air','m2',11.0,512,'wifi'): 1349000, ('air','m2',11.0,1024,'wifi'): 1649000,
    ('air','m2',11.0,128,'cellular'): 1129000, ('air','m2',11.0,256,'cellular'): 1279000,
    ('air','m2',11.0,512,'cellular'): 1579000, ('air','m2',11.0,1024,'cellular'): 1879000,
    ('air','m2',13.0,128,'wifi'): 1199000, ('air','m2',13.0,256,'wifi'): 1349000,
    ('air','m2',13.0,512,'wifi'): 1649000, ('air','m2',13.0,1024,'wifi'): 1949000,
    ('air','m2',13.0,128,'cellular'): 1429000, ('air','m2',13.0,256,'cellular'): 1579000,
    ('air','m2',13.0,512,'cellular'): 1879000, ('air','m2',13.0,1024,'cellular'): 2179000,
    ('air','m3',11.0,128,'wifi'): 949000, ('air','m3',11.0,256,'wifi'): 1099000,
    ('air','m3',11.0,512,'wifi'): 1399000, ('air','m3',11.0,1024,'wifi'): 1699000,
    ('air','m3',11.0,128,'cellular'): 1199000, ('air','m3',11.0,256,'cellular'): 1349000,
    ('air','m3',11.0,512,'cellular'): 1649000, ('air','m3',11.0,1024,'cellular'): 1949000,
    ('air','m3',13.0,128,'wifi'): 1249000, ('air','m3',13.0,256,'wifi'): 1399000,
    ('air','m3',13.0,512,'wifi'): 1699000, ('air','m3',13.0,1024,'wifi'): 1999000,
    ('air','m3',13.0,128,'cellular'): 1499000, ('air','m3',13.0,256,'cellular'): 1649000,
    ('air','m3',13.0,512,'cellular'): 1949000, ('air','m3',13.0,1024,'cellular'): 2249000,
    # BASIC
    ('basic','5세대',9.7,32,'wifi'): 430000, ('basic','5세대',9.7,128,'wifi'): 550000,
    ('basic','5세대',9.7,32,'cellular'): 600000, ('basic','5세대',9.7,128,'cellular'): 720000,
    ('basic','6세대',9.7,32,'wifi'): 430000, ('basic','6세대',9.7,128,'wifi'): 550000,
    ('basic','6세대',9.7,32,'cellular'): 600000, ('basic','6세대',9.7,128,'cellular'): 720000,
    ('basic','7세대',10.2,32,'wifi'): 449000, ('basic','7세대',10.2,128,'wifi'): 579000,
    ('basic','7세대',10.2,32,'cellular'): 619000, ('basic','7세대',10.2,128,'cellular'): 749000,
    ('basic','8세대',10.2,32,'wifi'): 449000, ('basic','8세대',10.2,128,'wifi'): 579000,
    ('basic','8세대',10.2,32,'cellular'): 619000, ('basic','8세대',10.2,128,'cellular'): 749000,
    ('basic','9세대',10.2,64,'wifi'): 449000, ('basic','9세대',10.2,256,'wifi'): 639000,
    ('basic','9세대',10.2,64,'cellular'): 619000, ('basic','9세대',10.2,256,'cellular'): 809000,
    ('basic','10세대',10.9,64,'wifi'): 679000, ('basic','10세대',10.9,256,'wifi'): 919000,
    ('basic','10세대',10.9,64,'cellular'): 919000, ('basic','10세대',10.9,256,'cellular'): 1159000,
    ('basic','11세대',11.0,128,'wifi'): 529000, ('basic','11세대',11.0,256,'wifi'): 679000,
    ('basic','11세대',11.0,512,'wifi'): 979000,
    ('basic','11세대',11.0,128,'cellular'): 779000, ('basic','11세대',11.0,256,'cellular'): 929000,
    ('basic','11세대',11.0,512,'cellular'): 1229000,
    # MINI
    ('mini','2세대',7.9,16,'wifi'): 420000, ('mini','2세대',7.9,32,'wifi'): 540000,
    ('mini','2세대',7.9,64,'wifi'): 660000, ('mini','2세대',7.9,128,'wifi'): 780000,
    ('mini','2세대',7.9,16,'cellular'): 570000, ('mini','2세대',7.9,32,'cellular'): 690000,
    ('mini','2세대',7.9,64,'cellular'): 810000, ('mini','2세대',7.9,128,'cellular'): 930000,
    ('mini','4세대',7.9,16,'wifi'): 480000, ('mini','4세대',7.9,32,'wifi'): 570000,
    ('mini','4세대',7.9,64,'wifi'): 600000, ('mini','4세대',7.9,128,'wifi'): 720000,
    ('mini','4세대',7.9,16,'cellular'): 630000, ('mini','4세대',7.9,32,'cellular'): 680000,
    ('mini','4세대',7.9,64,'cellular'): 750000, ('mini','4세대',7.9,128,'cellular'): 870000,
    ('mini','5세대',7.9,64,'wifi'): 499000, ('mini','5세대',7.9,256,'wifi'): 699000,
    ('mini','5세대',7.9,64,'cellular'): 669000, ('mini','5세대',7.9,256,'cellular'): 869000,
    ('mini','6세대',8.3,64,'wifi'): 649000, ('mini','6세대',8.3,256,'wifi'): 849000,
    ('mini','6세대',8.3,64,'cellular'): 849000, ('mini','6세대',8.3,256,'cellular'): 1049000,
    ('mini','7세대',8.3,128,'wifi'): 749000, ('mini','7세대',8.3,256,'wifi'): 899000,
    ('mini','7세대',8.3,512,'wifi'): 1199000,
    ('mini','7세대',8.3,128,'cellular'): 999000, ('mini','7세대',8.3,256,'cellular'): 1149000,
    ('mini','7세대',8.3,512,'cellular'): 1449000,
}

@st.cache_resource
def load_ipad_model():
    import joblib
    d = joblib.load('models/ipad_dep_model.pkl')
    return d['model'], d['CAT_MAP_DEP'], d['GEN_ORDER'], d['SRC_MAP'], d['features'], d['valid'], d['smooth_curves']

model, CAT_MAP_DEP, GEN_ORDER, SRC_MAP, features, valid_data, smooth_curves = load_ipad_model()

# ===== 현재시세 예측 모델 (모델 2) =====
@st.cache_resource
def load_current_price_model():
    import joblib
    d = joblib.load('models/ipad_cp_model.pkl')
    return d['model'], d['cp_features'], d['CAT_MAP'], d['CONN_MAP']

cp_model, cp_features, CAT_MAP, CONN_MAP = load_current_price_model()

# 애플케어 프리미엄 (카테고리별)
APPLECARE_PREMIUM = {'pro': 137888, 'mini': 60557, 'air': 40357, 'basic': 46985}

# 악세사리 프리미엄
PENCIL_PRO_PREMIUM = 103506
MAGIC_KB_PREMIUM = 78588

# 매입 할인율 (시장가 대비, RFP 기준 10~20% 범위)
# 근거: 위플레닛 실제 마진율 패턴(고가 모델 < 저가 모델) 참고하여 RFP 가이드라인에 맞춰 조정
BUY_DISCOUNT_RATE = {'pro': 0.10, 'air': 0.13, 'mini': 0.17, 'basic': 0.20}

def predict_hybrid(model, input_data, launch_price, category, generation, valid_data, smooth_curves):
    year_range = np.arange(0.5, 8.5, 0.5)
    gen_data = valid_data[(valid_data['category_re'] == category) & (valid_data['generation'] == generation)]
    max_year = gen_data['경과년수'].max() if len(gen_data) > 0 else 0

    prices = []
    last_rf_price = None
    last_rf_year = None

    for yr in year_range:
        inp = input_data.copy()
        inp['경과년수'] = yr

        if yr <= max_year + 0.5:
            pred_ratio = model.predict(inp)[0]
            pred = pred_ratio * launch_price
            last_rf_price = pred
            last_rf_year = yr
            prices.append(pred)
        else:
            curve_func = smooth_curves.get(category)
            if curve_func is not None and last_rf_price is not None:
                ref_at_anchor = float(curve_func(last_rf_year))
                ref_at_current = float(curve_func(yr))
                if ref_at_anchor > 0:
                    decay = ref_at_current / ref_at_anchor
                    pred = last_rf_price * max(decay, 0.1)
                    prices.append(max(pred, 0))
                else:
                    prices.append(last_rf_price * 0.9)
            else:
                prices.append(last_rf_price * 0.9 if last_rf_price else launch_price * 0.3)

    # 단조감소 강제
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            prices[i] = prices[i-1]

    return prices

# 출시연도 테이블
LAUNCH_YEARS = {
    'pro': {'1세대': 2015, '2세대': 2017, '3세대': 2018, '4세대': 2020, 'm1': 2021, 'm2': 2022, 'm4': 2024, 'm5': 2025},
    'air': {'1세대': 2013, '2세대': 2014, '3세대': 2019, '4세대': 2020, 'm1': 2022, 'm2': 2024, 'm3': 2025},
    'basic': {'5세대': 2017, '6세대': 2018, '7세대': 2019, '8세대': 2020, '9세대': 2021, '10세대': 2022, '11세대': 2025},
    'mini': {'2세대': 2013, '4세대': 2015, '5세대': 2019, '6세대': 2021, '7세대': 2024},
}

# 타이틀
st.title("중고 애플 기기 가격 예측")

# ===== 사이드바: 기기 선택 + 입력 =====
with st.sidebar:
    st.header("기기 선택")
    device = st.selectbox("기기", ['아이폰', '아이패드', '맥북'], key='device_select')

    if device == '아이패드':
        st.markdown("---")
        st.header("스펙 입력")

        cat_labels = {'pro': '아이패드 프로', 'air': '아이패드 에어', 'basic': '아이패드 (일반)', 'mini': '아이패드 미니'}
        cat_keys = list(VALID_COMBOS.keys())
        category = st.selectbox("카테고리", cat_keys, format_func=lambda x: cat_labels.get(x, x))

        gen_list = list(VALID_COMBOS[category].keys())
        generation = st.selectbox("세대", gen_list)

        combo = VALID_COMBOS[category][generation]
        size_list = combo['sizes']
        size = st.selectbox("화면 크기", size_list, format_func=lambda x: f"{x}인치")

        storage_list = combo['storages']
        storage_labels = {s: f"{int(s)}GB" if s < 1024 else f"{int(s/1024)}TB" for s in storage_list}
        storage = st.selectbox("용량", storage_list, format_func=lambda x: storage_labels[x])

        connectivity = st.radio("연결방식", ['wifi', 'cellular'], format_func=lambda x: 'Wi-Fi' if x == 'wifi' else 'Cellular')

        years = st.slider("출시 후 경과 년수", 0.5, 8.0, 2.0, 0.5)

        # ===== 상태 입력 토글 =====
        st.markdown("---")
        show_current = st.checkbox("상태 상세 입력")

        if show_current:
            st.header("상태 입력")
            cp_미개봉 = st.radio("미개봉 여부", ['아니오', '예'], key='cp_미개봉')

            if cp_미개봉 == '예':
                cp_battery = 100
                cp_기스 = '없음'
                cp_찍힘 = '없음'
                cp_풀박스 = '풀박스'
                st.info("미개봉 선택 시 배터리 100%, 기스 없음, 찍힘 없음, 풀박스로 자동 설정됩니다.")
                cp_애플케어 = st.radio("애플케어", ['없음', '있음'], key='cp_애플케어')
            else:
                cp_battery = st.slider("배터리 효율 (%)", 50, 100, 90, key='cp_battery')
                cp_기스 = st.selectbox("기스 상태", ['없음', '미세기스', '생활기스', '기스 심함'], key='cp_기스')
                cp_찍힘 = st.selectbox("찍힘 상태", ['없음', '약간', '심각'], key='cp_찍힘')
                cp_애플케어 = st.radio("애플케어", ['없음', '있음'], key='cp_애플케어')
                cp_풀박스 = st.selectbox("풀박스 상태", ['풀박스', '박스있음', '본체만'], key='cp_풀박스')

            st.markdown("---")
            st.header("악세사리 포함")
            # Pencil Pro: M2 이상만 선택 가능
            pencil_pro_compatible = generation in ['m2', 'm3', 'm4', 'm5']
            if pencil_pro_compatible:
                cp_pencil_pro = st.checkbox("Apple Pencil Pro 포함", key='cp_pencil_pro')
            else:
                cp_pencil_pro = False
                st.checkbox("Apple Pencil Pro 포함", value=False, disabled=True, key='cp_pencil_pro_disabled')
                st.caption("⚠️ Pencil Pro는 M2 이상 모델에서만 호환됩니다")
            # 매직키보드: 전체 선택 가능
            cp_magic_kb = st.checkbox("매직키보드 포함", key='cp_magic_kb')

        # ===== 마진 계산 토글 =====
        st.markdown("---")
        show_margin = st.checkbox("마진 시뮬레이션")

        if show_margin:
            st.header("마진 시뮬레이션")
            st.caption("매입 예상가는 자동 산출됩니다")

# ===== 메인 화면 =====
if device == '아이패드':
    st.info("번개장터·중고나라 실거래 데이터 4,325건 기반 **Hybrid 감가상각 모델** + **상태 반영 가격 보정 모델**이 적용된 매입가 자동 산정 시스템입니다.")

    # --- 출시가 계산 ---
    price_key = (category, generation, size, storage, connectivity)
    if price_key in LAUNCH_PRICES:
        launch_price = LAUNCH_PRICES[price_key]
    else:
        alt_key = (category, generation, size, storage, 'wifi')
        if alt_key in LAUNCH_PRICES:
            launch_price = LAUNCH_PRICES[alt_key] + 200000
        else:
            launch_price = 1000000
            st.warning("해당 조합의 정확한 출시가를 찾을 수 없습니다. 추정치를 사용합니다.")

    # --- 감가상각 인코딩 (수동 매핑, OOV는 최신값) ---
    gen_encoded = GEN_ORDER.get(generation, max(GEN_ORDER.values()))
    cat_encoded = CAT_MAP_DEP.get(category, max(CAT_MAP_DEP.values()))
    src_encoded = SRC_MAP.get('번개장터', 2)
    conn_encoded = 1 if connectivity == 'cellular' else 0

    input_data = pd.DataFrame([{
        '경과년수': years,
        'storage_gb': storage,
        'size': size,
        '출시가': launch_price,
        'gen_encoded': gen_encoded,
        'conn_encoded': conn_encoded,
        'cat_encoded': cat_encoded,
        'src_encoded': src_encoded,
    }])

    year_range = np.arange(0.5, 8.5, 0.5)
    all_prices = predict_hybrid(model, input_data, launch_price, category,
                                generation, valid_data, smooth_curves)

    year_idx = list(year_range).index(years) if years in year_range else 0
    predicted_price = all_prices[year_idx]
    residual_rate = (predicted_price / launch_price) * 100

    # ===== 1. 감가상각 시세 =====
    st.subheader("📉 1. 감가상각 시세")
    st.caption("Hybrid Interpolation 모델 — 세대별 실거래 패턴 + 스플라인 보간 기반 감가 곡선 예측")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("예상 중고가", f"{predicted_price:,.0f}원")
    with col_b:
        st.metric("출시가", f"{launch_price:,.0f}원")
    with col_c:
        st.metric("잔존가치율", f"{residual_rate:.1f}%  (▼{100-residual_rate:.1f}%)")

    gen_data = valid_data[(valid_data['category_re'] == category) & (valid_data['generation'] == generation)]
    max_year = gen_data['경과년수'].max() if len(gen_data) > 0 else 0
    if years > max_year + 0.5:
        st.warning(f"{generation}의 실제 데이터는 {max_year:.1f}년까지 존재합니다. 이후는 같은 라인업의 과거 모델 패턴을 참조한 추정치입니다.")

    chart_df = pd.DataFrame({
        '경과년수': year_range,
        '예상 중고가 (만원)': [p / 10000 for p in all_prices]
    })
    st.line_chart(chart_df.set_index('경과년수')['예상 중고가 (만원)'])

    if max_year + 0.5 < 8.0:
        st.caption(f"{max_year + 0.5:.1f}년 이후는 같은 카테고리의 과거 모델 감가상각 패턴을 참조한 추정치입니다.")

    st.info(f"**{cat_labels[category]} {generation} {size}인치 {storage_labels[storage]} {'Wi-Fi' if connectivity == 'wifi' else 'Cellular'}** | 경과 {years}년 | **{predicted_price:,.0f}원** (잔존가치 {residual_rate:.1f}%)")

    # ===== 2. 현재시세 예측 =====
    if show_current:
        st.markdown("---")
        st.subheader("💰 2. 현재시세 예측")
        st.caption("상태 보정 모델 — 배터리·기스·찍힘·애플케어·악세서리 등 개별 상태를 반영한 현재 시장가 예측")

        launch_year = LAUNCH_YEARS.get(category, {}).get(generation, 2020)
        launch_date = f"{launch_year}-06-01"
        경과년수_현재 = (CURRENT_DATE - pd.to_datetime(launch_date)).days / 365.25

        gen_num_match = re.search(r'(\d+)', generation)
        gen_num = float(gen_num_match.group(1)) if gen_num_match else 1.0

        기스_map = {'없음': 0, '미세기스': 1, '생활기스': 2, '기스 심함': 3}
        풀박스_map = {'풀박스': 2, '박스있음': 1, '본체만': 0}

        cp_input = pd.DataFrame([{
            '경과년수_현재': 경과년수_현재,
            '출시가': launch_price,
            'cat_encoded': CAT_MAP[category],
            'gen_encoded': gen_num,
            'conn_encoded': 1 if connectivity == 'cellular' else 0,
            'storage_gb': storage,
            'battery_final': cp_battery,
            '미개봉': 1 if cp_미개봉 == '예' else 0,
            '기스': 기스_map[cp_기스],
            '풀박스': 풀박스_map[cp_풀박스],
        }])

        감가액 = cp_model.predict(cp_input[cp_features])[0]

        # 룰 기반 조정: 찍힘 (위플레닛 정액 → 정률 변환)
        # 위플레닛 기준: 약간 -5만 / 심각 -15만 (전체 평균 68만 대비 7.4% / 22.1%)
        # → 각 카테고리 평균 판매가에 동일 비율 적용
        찍힘_카테고리 = {
            'pro': {'없음': 0, '약간': -77000, '심각': -230000},
            'air': {'없음': 0, '약간': -50000, '심각': -150000},
            'mini': {'없음': 0, '약간': -34000, '심각': -103000},
            'basic': {'없음': 0, '약간': -20000, '심각': -59000},
        }
        찍힘_차감 = 찍힘_카테고리.get(category, {'없음': 0, '약간': -50000, '심각': -150000})[cp_찍힘]


        # 룰 기반 조정: 애플케어
        애플케어_프리미엄 = APPLECARE_PREMIUM.get(category, 0) if cp_애플케어 == '있음' else 0

        # 악세사리 프리미엄
        pencil_pro_프리미엄 = PENCIL_PRO_PREMIUM if (show_current and cp_pencil_pro) else 0
        magic_kb_프리미엄 = MAGIC_KB_PREMIUM if (show_current and cp_magic_kb) else 0

        # 최종 가격 = 출시가 - 감가액 + 찍힘차감 + 애플케어 + 악세사리
        current_price = max(int(launch_price - 감가액 + 찍힘_차감 + 애플케어_프리미엄 + pencil_pro_프리미엄 + magic_kb_프리미엄), 0)
        current_residual = (current_price / launch_price) * 100
        diff = current_price - predicted_price

        # 매입 예상가 산출
        discount_rate = BUY_DISCOUNT_RATE.get(category, 0.15)
        buy_estimate = max(int(current_price * (1 - discount_rate)), 0)

        col_cp1, col_cp2, col_cp3, col_cp4 = st.columns(4)
        with col_cp1:
            st.metric("시장 판매 예상가", f"{current_price:,.0f}원")
        with col_cp2:
            st.metric("매입 예상가", f"{buy_estimate:,.0f}원")
        with col_cp3:
            st.metric("감가상각 시세", f"{predicted_price:,.0f}원")
        with col_cp4:
            st.metric("매입 할인율", f"{discount_rate*100:.0f}%")

        today_str = CURRENT_DATE.strftime('%Y.%m.%d')
        st.caption(f"경과년수: {경과년수_현재:.1f}년 (출시 {launch_year}년 → {today_str} 기준 자동 계산)")
        st.caption(f"매입 할인율 근거: RFP 가이드라인(10~20%) + 위플레닛 실제 매입가 패턴 참고")

    # ===== 3. 마진 시뮬레이션 =====
    if show_margin:
        st.markdown("---")
        st.subheader("📊 3. 마진 시뮬레이션")
        st.caption("매입 전략별(보수/기본/공격) 할인율 적용 + 운영비·재고 기간 리스크를 반영한 마진 자동 산출")

        # 판매 예상가 결정
        if show_current:
            sell_price = current_price
            sell_label = "현재시세 기준"
        else:
            sell_price = predicted_price
            sell_label = "감가상각 시세 기준"

        # 매입 전략 시나리오 선택 (발표자료 기준)
        # 공격=재고확보(매입가 높음), 보수=마진우선(매입가 낮음)
        SCENARIO = {
            '공격 (재고확보)': {'pro': 0.08, 'air': 0.10, 'mini': 0.13, 'basic': 0.15},
            '기본 (균형)': {'pro': 0.10, 'air': 0.13, 'mini': 0.17, 'basic': 0.20},
            '보수 (마진우선)': {'pro': 0.12, 'air': 0.15, 'mini': 0.20, 'basic': 0.25},
        }
        scenario_choice = st.selectbox("매입 전략", list(SCENARIO.keys()), index=1, key='scenario')
        scenario_discount = SCENARIO[scenario_choice].get(category, 0.15)

        # 시나리오 기반 매입가 계산
        기본매입가 = int(sell_price * (1 - scenario_discount))

        # 최소 마진 5% 보장
        최소마진_매입상한 = int(sell_price * 0.95)
        최종매입가 = min(기본매입가, 최소마진_매입상한)

        # 부대비용 (사용자 입력 + 배송비 고정)
        운영비_m = st.number_input("운영비 (원)", min_value=0, value=30000, step=5000, key='운영비_m')
        repair_cost_m = st.number_input("수리비 (원)", min_value=0, value=0, step=10000, key='repair_cost_m')
        shipping_cost_m = 8000  # 배송비 고정

        # 총 비용 = 매입가 + 운영비 + 수리비 + 배송비
        total_cost = 최종매입가 + 운영비_m + repair_cost_m + shipping_cost_m

        # 마진 = 판매 예상가 - 총 비용
        순마진 = sell_price - total_cost
        순마진율 = (순마진 / sell_price * 100) if sell_price > 0 else 0

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("최종 매입가", f"{최종매입가:,.0f}원")
        with col_m2:
            st.metric("판매 예상가", f"{sell_price:,.0f}원", help=sell_label)
        with col_m3:
            st.metric("마진", f"{순마진:,.0f}원  ({순마진율:.1f}%)")

        # 비용 상세 내역
        st.markdown("📋 **비용 상세**")
        cost_df = pd.DataFrame({
            '항목': ['매입가', f'매입 차감 ({scenario_discount*100:.0f}%)',
                    '운영비', '수리비', '배송비 (고정 8,000원)', '총 비용',
                    '판매 예상가', '마진'],
            '금액': [f"{최종매입가:,}원", f"-{int(sell_price * scenario_discount):,}원",
                    f"{운영비_m:,}원",
                    f"{repair_cost_m:,}원", f"{shipping_cost_m:,}원", f"{total_cost:,}원",
                    f"{sell_price:,}원", f"{순마진:,}원 ({순마진율:.1f}%)"]
        })
        st.table(cost_df)

        # 재고 기간별 시뮬레이션 (위플레닛 실제 재고 데이터 기반)
        st.markdown("**📦 재고 기간별 가격 하락 시뮬레이션**")
        st.caption("위플레닛 실제 데이터: 평균 재고 359일, 50.4%가 1년 이상 재고")
        판매가_즉시 = sell_price
        판매가_보통 = int(sell_price * 0.95)
        판매가_장기 = int(sell_price * 0.90)
        마진_즉시 = 판매가_즉시 - total_cost
        마진_보통 = 판매가_보통 - total_cost
        마진_장기 = 판매가_장기 - total_cost
        마진율_즉시 = (마진_즉시 / 판매가_즉시 * 100) if 판매가_즉시 > 0 else 0
        마진율_보통 = (마진_보통 / 판매가_보통 * 100) if 판매가_보통 > 0 else 0
        마진율_장기 = (마진_장기 / 판매가_장기 * 100) if 판매가_장기 > 0 else 0

        inv_df = pd.DataFrame({
            '기간': ['빠른 판매 (~6개월, 25.4%)', '보통 (6개월~1년, 24.1%, -5%)', '장기 재고 (1년+, 50.4%, -10%)'],
            '판매가': [f"{판매가_즉시:,}원", f"{판매가_보통:,}원", f"{판매가_장기:,}원"],
            '마진': [f"{마진_즉시:,}원", f"{마진_보통:,}원", f"{마진_장기:,}원"],
            '마진율': [f"{마진율_즉시:.1f}%", f"{마진율_보통:.1f}%", f"{마진율_장기:.1f}%"]
        })
        st.table(inv_df)

        st.caption(f"매입 전략: {scenario_choice} | 매입 차감 {scenario_discount*100:.0f}% | 최소 마진 5% 보장")

elif device == '아이폰':
    pass  # 아이폰 탭은 아래에서 처리

elif device == "맥북":
    # =========================================================
    # Cell 1: 유틸 함수 (로직 그대로)
    # =========================================================
    def convert_to_gb(x):
        if pd.isna(x):
            return np.nan
        x = str(x).lower().strip()
        if 'tb' in x:
            return float(x.replace('tb', '')) * 1024
        if 'gb' in x:
            return float(x.replace('gb', ''))
        return np.nan

    def extract_chip_info(chip_name):
        if pd.isna(chip_name):
            return 1, 1
        chip_name = str(chip_name).lower().replace(" ", "")
        gen_match = re.search(r'm(\d+)', chip_name)
        gen = int(gen_match.group(1)) if gen_match else 1
        tier = 1
        if 'pro' in chip_name:
            tier = 2
        elif 'max' in chip_name:
            tier = 3
        return gen, tier

    def ram_bucket(x):
        if pd.isna(x):
            return np.nan
        if x <= 8: return 0
        elif x <= 16: return 1
        elif x <= 24: return 2
        elif x <= 32: return 3
        elif x <= 48: return 4
        elif x <= 64: return 5
        else: return 6

    def get_battery_group(battery_health):
        """마진 시뮬레이션용 (노트북 원본 매핑)"""
        if pd.isna(battery_health):
            return np.nan
        if battery_health <= 80: return "80이하"
        elif battery_health <= 85: return "80~85"
        elif battery_health <= 90: return "85~90"
        elif battery_health <= 95: return "90~95"
        elif battery_health <= 98: return "95~98"
        else: return "98~100"

    def get_battery_group_xgb(battery_health):
        """맥북 XGB 모델용 (save_models.py 학습 시 매핑)"""
        if pd.isna(battery_health):
            return np.nan
        if battery_health >= 90: return "high"
        if battery_health >= 80: return "mid"
        return "low"

    def normalize_model_type(model_value):
        if pd.isna(model_value): return "unknown"
        m = str(model_value).lower().strip()
        if "air" in m: return "air"
        elif "pro" in m: return "pro"
        else: return "unknown"

    def get_size_group(model_type, inch):
        if pd.isna(inch): return "unknown"
        inch = float(inch)
        if model_type == "air":
            return "small" if inch <= 14 else "large"
        elif model_type == "pro":
            return "small" if inch <= 14 else "large"
        return "unknown"

    def calc_accessory_bonus(row):
        mouse_num = int(pd.to_numeric(row.get("mouse_num", 0), errors="coerce") if not pd.isna(row.get("mouse_num", 0)) else 0)
        keyboard_num = int(pd.to_numeric(row.get("keyboard_num", 0), errors="coerce") if not pd.isna(row.get("keyboard_num", 0)) else 0)
        pad_num = int(pd.to_numeric(row.get("pad_num", 0), errors="coerce") if not pd.isna(row.get("pad_num", 0)) else 0)
        bonus = 30000 * mouse_num + 50000 * keyboard_num + 80000 * pad_num
        return min(bonus, 120000)

    APPLECARE_BONUS_TABLE = {
        ("air", "small"): {"mid": 30000, "high": 50000, "max": 70000},
        ("air", "large"): {"mid": 30000, "high": 60000, "max": 90000},
        ("pro", "small"): {"mid": 50000, "high": 90000, "max": 130000},
        ("pro", "large"): {"mid": 50000, "high": 100000, "max": 150000}
    }

    def calc_applecare_bonus(row):
        applecare = int(pd.to_numeric(row.get("applecare", 0), errors="coerce") if not pd.isna(row.get("applecare", 0)) else 0)
        care_num = pd.to_numeric(row.get("care_num", np.nan), errors="coerce")
        if applecare != 1 or pd.isna(care_num): return 0
        model_type = normalize_model_type(row.get("model", "unknown"))
        size_group = get_size_group(model_type, row.get("inch", np.nan))
        if model_type == "unknown" or size_group == "unknown": return 0
        effective_months = max(int(care_num) - 3, 0)
        table = APPLECARE_BONUS_TABLE.get((model_type, size_group), None)
        if table is None: return 0
        if 9 <= effective_months <= 20: return table["mid"]
        elif 21 <= effective_months <= 32: return table["high"]
        elif effective_months >= 33: return table["max"]
        else: return 0

    # 매입가 산정 함수 (Cell 1)
    def state_sale_premium_adjustment(row):
        adj = 0.0
        battery_group = get_battery_group(row.get("battery_health", np.nan))
        fault = str(row.get("fault_level", "unknown")).strip()
        unseal = int(row.get("unseal", 0))
        fullbox = int(row.get("fullbox", 0))
        if unseal == 1: adj += 0.03
        if fullbox == 1: adj += 0.01
        if battery_group == "98~100": adj += 0.01
        elif battery_group == "95~98": adj += 0.005
        elif battery_group == "85~90": adj -= 0.01
        elif battery_group == "80~85": adj -= 0.02
        elif battery_group == "80이하": adj -= 0.04
        if fault == "정상": adj += 0.01
        elif fault in ["미세사용감", "사용감"]: adj -= 0.01
        elif fault in ["외관하자", "외관 하자"]: adj -= 0.03
        elif fault in ["액정결함", "액정 결함"]: adj -= 0.06
        elif fault in ["기능고장", "기능 고장"]: adj -= 0.08
        return adj

    def state_safety_margin_adjustment(row):
        adj = 0.0
        battery_group = get_battery_group(row.get("battery_health", np.nan))
        fault = str(row.get("fault_level", "unknown")).strip()
        unseal = int(row.get("unseal", 0))
        fullbox = int(row.get("fullbox", 0))
        if battery_group == "95~98": adj += 0.005
        elif battery_group == "90~95": adj += 0.01
        elif battery_group == "85~90": adj += 0.02
        elif battery_group == "80~85": adj += 0.03
        elif battery_group == "80이하": adj += 0.05
        if fault == "정상": adj += 0.00
        elif fault in ["미세사용감"]: adj += 0.005
        elif fault in ["사용감"]: adj += 0.01
        elif fault in ["외관하자", "외관 하자"]: adj += 0.03
        elif fault in ["액정결함", "액정 결함"]: adj += 0.06
        elif fault in ["기능고장", "기능 고장"]: adj += 0.08
        if unseal == 1: adj -= 0.01
        if fullbox == 1: adj -= 0.005
        return max(adj, 0.0)

    SCENARIOS = {
        "보수": {"base_sale_premium": 0.05, "op_cost_rate": 0.10, "target_margin_rate": 0.12, "base_safety_margin": 0.05},
        "표준": {"base_sale_premium": 0.07, "op_cost_rate": 0.10, "target_margin_rate": 0.10, "base_safety_margin": 0.03},
        "공격": {"base_sale_premium": 0.10, "op_cost_rate": 0.09, "target_margin_rate": 0.08, "base_safety_margin": 0.02}
    }

    def calculate_purchase_offer(row_dict, scenario_name="표준", repair_cost=0):
        s = SCENARIOS[scenario_name]
        predicted_price = float(row_dict["predicted_price"])
        sale_premium = s["base_sale_premium"] + state_sale_premium_adjustment(row_dict)
        safety_margin = s["base_safety_margin"] + state_safety_margin_adjustment(row_dict)
        sale_premium = min(max(sale_premium, 0.00), 0.15)
        safety_margin = min(max(safety_margin, 0.00), 0.15)
        target_sale_price = predicted_price * (1 + sale_premium)
        total_deduction_rate = s["op_cost_rate"] + s["target_margin_rate"] + safety_margin
        total_deduction_rate = min(max(total_deduction_rate, 0.0), 0.50)
        final_purchase_price = target_sale_price * (1 - total_deduction_rate)
        expected_gross_spread = target_sale_price - final_purchase_price - repair_cost
        implied_purchase_ratio = final_purchase_price / predicted_price
        return {
            "scenario": scenario_name,
            "predicted_price": round(predicted_price),
            "sale_premium_rate": round(sale_premium, 4),
            "target_sale_price": round(target_sale_price),
            "op_cost_rate": round(s["op_cost_rate"], 4),
            "target_margin_rate": round(s["target_margin_rate"], 4),
            "safety_margin_rate": round(safety_margin, 4),
            "total_deduction_rate": round(total_deduction_rate, 4),
            "final_purchase_price": round(final_purchase_price),
            "repair_cost": round(repair_cost),
            "expected_gross_spread": round(expected_gross_spread),
            "implied_purchase_ratio_vs_pred": round(implied_purchase_ratio, 4)
        }

    # =========================================================
    # Cell 3: 데이터 테이블 + 매칭 함수 (로직 그대로)
    # =========================================================
    MODEL_MATCH_DATA = [
        [2020, "MacBook Air", 13, "M1", "8GB", "256GB", 1290000],
        [2020, "MacBook Pro", 13, "M1", "8GB", "256GB", 1690000],
        [2021, "MacBook Pro", 14, "M1Max", "32GB", "512GB", 3360000],
        [2021, "MacBook Pro", 16, "M1Max", "32GB", "512GB", 3690000],
        [2021, "MacBook Pro", 14, "M1Pro", "16GB", "512GB", 2690000],
        [2021, "MacBook Pro", 16, "M1Pro", "16GB", "512GB", 3390000],
        [2022, "MacBook Air", 13, "M2", "8GB", "256GB", 1690000],
        [2022, "MacBook Pro", 13, "M2", "8GB", "256GB", 1790000],
        [2022, "MacBook Pro", 15, "M2", "8GB", "256GB", 1890000],
        [2023, "MacBook Air", 15, "M2", "8GB", "256GB", 1890000],
        [2023, "MacBook Pro", 14, "M2Max", "32GB", "512GB", 4290000],
        [2023, "MacBook Pro", 16, "M2Max", "32GB", "512GB", 4840000],
        [2023, "MacBook Pro", 14, "M2Pro", "16GB", "512GB", 2990000],
        [2023, "MacBook Pro", 16, "M2Pro", "16GB", "512GB", 3690000],
        [2024, "MacBook Air", 13, "M3", "8GB", "256GB", 1590000],
        [2024, "MacBook Air", 15, "M3", "8GB", "256GB", 1890000],
        [2024, "MacBook Pro", 14, "M3", "8GB", "512GB", 2390000],
        [2024, "MacBook Pro", 14, "M3Max", "36GB", "512GB", 3990000],
        [2024, "MacBook Pro", 16, "M3Max", "36GB", "512GB", 4690000],
        [2024, "MacBook Pro", 14, "M3Pro", "18GB", "512GB", 2790000],
        [2024, "MacBook Pro", 16, "M3Pro", "18GB", "512GB", 3390000],
        [2024, "MacBook Pro", 14, "M4", "16GB", "512GB", 2390000],
        [2024, "MacBook Pro", 14, "M4Max", "36GB", "1TB", 4090000],
        [2024, "MacBook Pro", 16, "M4Max", "36GB", "1TB", 4790000],
        [2024, "MacBook Pro", 14, "M4Pro", "24GB", "512GB", 2790000],
        [2024, "MacBook Pro", 16, "M4Pro", "24GB", "512GB", 3390000],
        [2025, "MacBook Air", 13, "M4", "16GB", "256GB", 1590000],
        [2025, "MacBook Air", 15, "M4", "16GB", "256GB", 1890000],
        [2025, "MacBook Pro", 14, "M5", "16GB", "512GB", 2590000],
        [2026, "MacBook Air", 13, "M5", "16GB", "512GB", 1590000],
        [2026, "MacBook Air", 15, "M5", "16GB", "512GB", 1790000],
        [2026, "MacBook Pro", 14, "M5Max", "36GB", "1TB", 5790000],
        [2026, "MacBook Pro", 16, "M5Max", "36GB", "1TB", 6290000],
        [2026, "MacBook Pro", 14, "M5Pro", "24GB", "1TB", 3490000],
        [2026, "MacBook Pro", 16, "M5Pro", "24GB", "1TB", 4290000],
    ]

    model_match_df = pd.DataFrame(
        MODEL_MATCH_DATA,
        columns=["release_year", "model", "inch", "chip", "release_ram", "release_ssd", "release_price"]
    )

    RAM_UPGRADE_TABLE = {
        "8GB":   {"8GB": 0, "16GB": 270000, "24GB": 540000},
        "16GB":  {"16GB": 0, "24GB": 270000, "32GB": 540000, "48GB": 810000, "64GB": 1080000},
        "18GB":  {"18GB": 0, "36GB": 540000},
        "24GB":  {"24GB": 0, "36GB": 540000, "48GB": 540000, "64GB": 810000, "128GB": 1890000},
        "32GB":  {"32GB": 0, "36GB": 540000},
        "36GB":  {"36GB": 0, "48GB": 540000, "64GB": 810000, "96GB": 1350000, "128GB": 1890000},
    }

    SSD_UPGRADE_TABLE = {
        "128GB": {"128GB": 0, "256GB": 150000, "512GB": 300000, "1TB": 570000},
        "256GB": {"256GB": 0, "512GB": 270000, "1TB": 540000, "2TB": 1080000, "4TB": 2160000},
        "512GB": {"512GB": 0, "1TB": 270000, "2TB": 810000, "4TB": 1890000, "8TB": 3510000},
        "1TB":   {"1TB": 0, "2TB": 540000, "4TB": 1620000, "8TB": 3240000},
    }

    FAULT_SCORE_MAP = {
        "미세기스": 1, "생활기스": 1,
        "스크래치": 2, "찌그러짐": 2,
        "깨짐": 4, "균열": 4,
        "액정파손": 8, "화면트러블": 8, "변색": 8, "화면깨짐": 8, "잔상": 8,
        "전원불가": 16, "침수": 16, "충전불량": 16, "키보드/패드 불량": 16,
    }

    def normalize_chip_name(chip):
        return chip.replace(" ", "").strip()

    def get_fault_level_from_score(score):
        if score == 0: return "정상"
        elif 1 <= score <= 3: return "사용감"
        elif 4 <= score <= 7: return "외관 하자"
        elif 8 <= score <= 15: return "액정 결함"
        else: return "기능 고장"

    def get_upgrade_price(base_spec, selected_spec, table):
        if base_spec not in table: return None
        return table[base_spec].get(selected_spec, None)

    def match_model_info(model, inch, chip):
        chip_norm = normalize_chip_name(chip)
        temp = model_match_df.copy()
        temp["chip_norm"] = temp["chip"].apply(normalize_chip_name)
        matched = temp[
            (temp["model"] == model) &
            (temp["inch"] == inch) &
            (temp["chip_norm"] == chip_norm)
        ]
        if matched.empty: return None
        return matched.iloc[0].to_dict()

    def build_macbook_input_dict(
        model, inch, chip, ram, storage, battery_health,
        selected_fault_items, unseal_checked, fullbox_checked,
        applecare_checked, care_num,
        magic_keyboard_checked, magic_trackpad_checked, magic_mouse_checked
    ):
        matched = match_model_info(model, inch, chip)
        if matched is None:
            return None, "해당 모델 조합의 출시정보를 찾을 수 없습니다. 스펙을 확인하세요."

        release_year = int(matched["release_year"])
        base_ram = matched["release_ram"]
        base_ssd = matched["release_ssd"]
        base_release_price = int(matched["release_price"])

        ram_upgrade_price = get_upgrade_price(base_ram, ram, RAM_UPGRADE_TABLE)
        ssd_upgrade_price = get_upgrade_price(base_ssd, storage, SSD_UPGRADE_TABLE)

        option_match_warning = None
        if ram_upgrade_price is None or ssd_upgrade_price is None:
            option_match_warning = "업그레이드 금액 매칭 안됨, 스펙을 확인하세요."

        final_release_price = base_release_price
        if ram_upgrade_price is not None: final_release_price += ram_upgrade_price
        if ssd_upgrade_price is not None: final_release_price += ssd_upgrade_price

        ram_upgraded = (ram != base_ram)
        storage_upgraded = (storage != base_ssd)

        unique_scores = set()
        for item in selected_fault_items:
            if item in FAULT_SCORE_MAP:
                unique_scores.add(FAULT_SCORE_MAP[item])
        fault_score = int(sum(unique_scores))
        fault_level = get_fault_level_from_score(fault_score)

        unseal = 1 if unseal_checked else 0
        fullbox = 1 if fullbox_checked else 0
        applecare = 1 if applecare_checked else 0
        care_num_val = int(care_num) if applecare == 1 else 0
        keyboard_num = 1 if magic_keyboard_checked else 0
        pad_num = 1 if magic_trackpad_checked else 0
        mouse_num = 1 if magic_mouse_checked else 0

        result = {
            "model": model, "inch": int(inch), "chip": chip,
            "ram": ram, "storage": storage,
            "release_year": release_year, "release_ram": base_ram,
            "release_ssd": base_ssd, "base_release_price": base_release_price,
            "release_price": final_release_price,
            "ram_upgrade_price": ram_upgrade_price, "ssd_upgrade_price": ssd_upgrade_price,
            "ram_upgraded": ram_upgraded, "storage_upgraded": storage_upgraded,
            "battery_health": int(battery_health), "fault_score": fault_score,
            "fault_level": fault_level,
            "unseal": unseal, "fullbox": fullbox,
            "applecare": applecare, "care_num": care_num_val,
            "keyboard_num": keyboard_num, "pad_num": pad_num, "mouse_num": mouse_num,
        }
        return result, option_match_warning

    # =========================================================
    # @st.cache_resource: RF + XGB 모델 학습 (Cell 1 로직 그대로)
    # =========================================================
    MAC_DATA_PATH = 'data/맥북/'

    @st.cache_resource
    def load_macbook_models():
        import joblib
        d = joblib.load('models/macbook_models.pkl')
        return d['rf_model'], d['rf_features'], d['rf_r2'], d['rf_mae'], d['xgb_model'], d['xgb_features']

    mac_rf_model, mac_rf_features, mac_rf_r2, mac_rf_mae, mac_xgb_model, mac_xgb_features = load_macbook_models()

    # =========================================================
    # 사이드바 입력 (아이패드 패턴)
    # =========================================================
    with st.sidebar:
        st.markdown("---")
        st.header("맥북 스펙 입력")

        mac_model = st.selectbox("모델", ["MacBook Pro", "MacBook Air"], key='mac_model')

        # 모델에 따라 인치 동적 필터링
        available_inches = sorted(model_match_df[model_match_df["model"] == mac_model]["inch"].unique().tolist())
        mac_inch = st.selectbox("크기", available_inches, format_func=lambda x: f"{x}인치", key='mac_inch')

        # 모델+인치에 따라 칩 동적 필터링
        available_chips = model_match_df[
            (model_match_df["model"] == mac_model) & (model_match_df["inch"] == mac_inch)
        ]["chip"].unique().tolist()
        mac_chip = st.selectbox("칩", available_chips, key='mac_chip')

        # 모델+인치+칩에 따라 기본 RAM/SSD 확인 후 업그레이드 옵션 제공
        matched_info = match_model_info(mac_model, mac_inch, mac_chip)
        if matched_info:
            base_ram = matched_info["release_ram"]
            base_ssd = matched_info["release_ssd"]
            # RAM: 기본값 이상의 옵션만 표시
            all_ram = ["8GB", "16GB", "18GB", "24GB", "32GB", "36GB", "48GB", "64GB", "96GB", "128GB"]
            if base_ram in RAM_UPGRADE_TABLE:
                available_ram = [base_ram] + [r for r in RAM_UPGRADE_TABLE[base_ram].keys() if r != base_ram]
            else:
                available_ram = [base_ram]
            mac_ram = st.selectbox("RAM", available_ram, key='mac_ram')

            # SSD: 기본값 이상의 옵션만 표시
            if base_ssd in SSD_UPGRADE_TABLE:
                available_ssd = [base_ssd] + [s for s in SSD_UPGRADE_TABLE[base_ssd].keys() if s != base_ssd]
            else:
                available_ssd = [base_ssd]
            mac_storage = st.selectbox("저장용량", available_ssd, key='mac_storage')

            st.caption(f"기본 사양: RAM {base_ram} / SSD {base_ssd} | 출시가 {matched_info['release_price']:,}원")
        else:
            mac_ram = st.selectbox("RAM", ["8GB", "16GB", "24GB", "32GB", "36GB", "48GB", "64GB"], key='mac_ram')
            mac_storage = st.selectbox("저장용량", ["256GB", "512GB", "1TB", "2TB", "4TB", "8TB"], key='mac_storage')

        # 상태 상세 입력 토글
        st.markdown("---")
        show_current_mac = st.checkbox("상태 상세 입력", key='show_current_mac')

        mac_battery = 95
        mac_fault_items = []
        mac_unseal = False
        mac_fullbox = False
        mac_applecare = False
        mac_care_num = 0
        mac_keyboard = False
        mac_trackpad = False
        mac_mouse = False

        if show_current_mac:
            st.header("상태 입력")
            mac_battery = st.slider("배터리 효율", 0, 100, 95, key='mac_battery')

            st.markdown("**하자 여부 (해당되는 항목 모두 체크)**")
            fault_items = list(FAULT_SCORE_MAP.keys())
            mac_fault_items = []
            cols = st.columns(3)
            for i, item in enumerate(fault_items):
                with cols[i % 3]:
                    checked = st.checkbox(item, key=f"mac_fault_{item}")
                    if checked:
                        mac_fault_items.append(item)

            st.markdown("**기타 상태 / 옵션**")
            mac_unseal = st.checkbox("미개봉", key='mac_unseal')
            mac_fullbox = st.checkbox("풀박스", key='mac_fullbox')
            mac_applecare = st.checkbox("애플케어 플러스", key='mac_applecare')
            mac_care_num = st.slider("애플케어 잔여 개월 수", 0, 36, 0, key='mac_care_num')

            st.markdown("**함께 판매할 악세서리**")
            mac_keyboard = st.checkbox("매직 키보드", key='mac_keyboard')
            mac_trackpad = st.checkbox("매직 트랙패드", key='mac_trackpad')
            mac_mouse = st.checkbox("매직 마우스", key='mac_mouse')

        # 마진 시뮬레이션 토글
        st.markdown("---")
        show_margin_mac = st.checkbox("마진 시뮬레이션", key='show_margin_mac')
        if show_margin_mac:
            mac_repair_cost = st.number_input("수리비용 (원)", min_value=0, value=0, step=10000, key='mac_repair_cost')
        else:
            mac_repair_cost = 0

    # =========================================================
    # 메인 화면: 입력 매칭 + 출시가 계산
    # =========================================================
    result_dict, warning_msg = build_macbook_input_dict(
        model=mac_model, inch=mac_inch, chip=mac_chip,
        ram=mac_ram, storage=mac_storage, battery_health=mac_battery,
        selected_fault_items=mac_fault_items,
        unseal_checked=mac_unseal, fullbox_checked=mac_fullbox,
        applecare_checked=mac_applecare, care_num=mac_care_num,
        magic_keyboard_checked=mac_keyboard,
        magic_trackpad_checked=mac_trackpad,
        magic_mouse_checked=mac_mouse
    )

    st.info("중고 거래 실데이터 3,301건 기반 **RandomForest 감가상각 모델**(R² 0.91) + **XGBoost 상태 보정 모델** + 룰 기반 옵션 가감이 적용된 매입가 자동 산정 시스템입니다.")

    if result_dict is None:
        st.error(warning_msg)
    else:
        release_price = result_dict["release_price"]
        release_year = result_dict["release_year"]

        if warning_msg:
            st.warning(warning_msg)

        # =========================================================
        # 섹션 1: 감가상각 시세 (RF 모델 예측)
        # =========================================================
        st.subheader("📉 1. 감가상각 시세")
        st.caption("RandomForest 모델 — 스펙(모델/칩/RAM/SSD) 기반 잔존가치율 예측 | R² 0.91, MAE 116,100원")

        # RF 입력 변환
        mac_ram_gb = convert_to_gb(mac_ram)
        mac_storage_gb = convert_to_gb(mac_storage)
        chip_gen, chip_tier = extract_chip_info(mac_chip)
        mac_age = 2026 - release_year

        rf_input = {
            "ram_level": ram_bucket(mac_ram_gb),
            "storage_gb": mac_storage_gb,
            "inch": mac_inch,
            "chip_gen": chip_gen,
            "chip_tier": chip_tier,
        }

        # model_ 더미 변수
        model_key = "model_" + mac_model.lower().replace(" ", "")
        for f in mac_rf_features:
            if f.startswith("model_"):
                rf_input[f] = 1 if f == model_key else 0

        # price_segment_ 더미 변수
        if release_price <= 1600000:
            seg = 0
        elif release_price <= 2000000:
            seg = 1
        elif release_price <= 3000000:
            seg = 2
        elif release_price <= 4500000:
            seg = 3
        else:
            seg = 4

        for f in mac_rf_features:
            if f.startswith("price_segment_"):
                seg_num = int(f.split("_")[-1])
                rf_input[f] = 1 if seg_num == seg else 0

        # 누락 feature 0으로 채우기
        for f in mac_rf_features:
            if f not in rf_input:
                rf_input[f] = 0

        rf_input_df = pd.DataFrame([rf_input])[mac_rf_features]
        rf_pred_log = mac_rf_model.predict(rf_input_df)[0]
        rf_pred_ratio = np.exp(rf_pred_log)
        rf_predicted_price = int(rf_pred_ratio * release_price)
        residual_rate = rf_pred_ratio * 100

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("예상 중고가", f"{rf_predicted_price:,.0f}원")
        with col_b:
            st.metric("출시가", f"{release_price:,.0f}원")
        with col_c:
            st.metric("잔존가치율", f"{residual_rate:.1f}%  (▼{100-residual_rate:.1f}%)")

        # 경과년수별 차트
        # RF 모델에 age feature가 없으므로, 현재 예측가를 기준으로
        # 연간 감가율을 적용하여 경과년수별 곡선 생성
        # (맥북 중고시장 평균 연간 감가율: chip_tier가 높을수록 가치 유지)
        annual_depreciation = {1: 0.13, 2: 0.11, 3: 0.09}  # chip_tier별 연간 감가율
        dep_rate = annual_depreciation.get(chip_tier, 0.13)

        year_prices = []
        for yr in range(0, 7):
            if yr <= mac_age:
                # 출시~현재: 현재 예측가 기준 역산
                ratio = rf_pred_ratio ** (yr / mac_age) if mac_age > 0 else 1.0
                price = int(release_price * ratio)
            else:
                # 미래: 현재 예측가에서 연간 감가율 적용
                years_ahead = yr - mac_age
                price = int(rf_predicted_price * ((1 - dep_rate) ** years_ahead))
            year_prices.append(price)

        # 0년차는 출시가
        year_prices[0] = release_price

        # 가격이 증가하지 않도록 보정
        for i in range(1, len(year_prices)):
            if year_prices[i] > year_prices[i-1]:
                year_prices[i] = year_prices[i-1]

        chart_df = pd.DataFrame({
            '경과년수': range(0, 7),
            '예상 중고가 (만원)': [p / 10000 for p in year_prices]
        })
        st.line_chart(chart_df.set_index('경과년수')['예상 중고가 (만원)'])

        st.info(f"**{mac_model} {mac_inch}인치 {mac_chip} RAM {mac_ram} {mac_storage}** | 출시 {release_year}년 | **{rf_predicted_price:,.0f}원** (잔존가치 {residual_rate:.1f}%)")

        # =========================================================
        # 섹션 2: 현재시세 예측 (XGB 보정 + 옵션)
        # =========================================================
        if show_current_mac:
            st.markdown("---")
            st.subheader("💰 2. 현재시세 예측")
            st.caption("XGBoost 보정 모델 — 배터리·하자·미개봉·풀박스 등 상태 요소 반영 + 악세서리·애플케어 옵션 가감 적용")

            # XGB 입력
            battery_group = get_battery_group_xgb(mac_battery)
            xgb_input = {
                "unseal": result_dict["unseal"],
                "fullbox": result_dict["fullbox"],
            }

            # battery_group 더미
            for f in mac_xgb_features:
                if f.startswith("battery_group_"):
                    group_name = f.replace("battery_group_", "")
                    xgb_input[f] = 1 if group_name == battery_group else 0

            # fault_level 더미
            fault_level = result_dict["fault_level"]
            for f in mac_xgb_features:
                if f.startswith("fault_level_"):
                    level_name = f.replace("fault_level_", "")
                    xgb_input[f] = 1 if level_name == fault_level else 0

            # 누락 feature 0으로 채우기
            for f in mac_xgb_features:
                if f not in xgb_input:
                    xgb_input[f] = 0

            xgb_input_df = pd.DataFrame([xgb_input])[mac_xgb_features]
            xgb_pred_log_ratio = mac_xgb_model.predict(xgb_input_df)[0]
            xgb_ratio = np.exp(xgb_pred_log_ratio)

            # 상태 반영 예측가 = RF 예측가 x XGB 보정
            predicted_price = int(rf_predicted_price * xgb_ratio)

            # 옵션 보너스 반영 (Cell 1 로직)
            accessory_bonus = calc_accessory_bonus(result_dict)
            applecare_bonus = calc_applecare_bonus(result_dict)
            option_bonus = accessory_bonus + applecare_bonus

            current_price = predicted_price + option_bonus
            current_residual = (current_price / release_price) * 100

            # 매입 할인율
            mac_discount_rate = 0.12 if "pro" in mac_model.lower() else 0.15
            buy_estimate = max(int(current_price * (1 - mac_discount_rate)), 0)

            col_cp1, col_cp2, col_cp3, col_cp4 = st.columns(4)
            with col_cp1:
                st.metric("시장 판매 예상가", f"{current_price:,.0f}원")
            with col_cp2:
                st.metric("매입 예상가", f"{buy_estimate:,.0f}원")
            with col_cp3:
                st.metric("감가상각 시세", f"{rf_predicted_price:,.0f}원")
            with col_cp4:
                st.metric("XGB 보정률", f"{xgb_ratio:.4f}")

            if option_bonus > 0:
                st.caption(f"옵션 보너스: 악세서리 {accessory_bonus:,}원 + 애플케어 {applecare_bonus:,}원 = {option_bonus:,}원")

            st.caption(f"배터리: {mac_battery}% | 하자: {fault_level} | 미개봉: {'예' if mac_unseal else '아니오'} | 풀박스: {'예' if mac_fullbox else '아니오'}")

        # =========================================================
        # 섹션 3: 마진 시뮬레이션
        # =========================================================
        if show_margin_mac:
            st.markdown("---")
            st.subheader("📊 3. 마진 시뮬레이션")
            st.caption("보수/표준/공격 3개 시나리오별 판매 프리미엄·운영비율·안전마진을 자동 적용하여 매입가 산정")

            # 판매 예상가 결정
            if show_current_mac:
                sell_price = current_price
                sell_label = "현재시세 기준"
            else:
                sell_price = rf_predicted_price
                sell_label = "감가상각 시세 기준"

            # 3개 시나리오 모두 자동 계산 (Cell 1 로직 그대로)
            offer_input = result_dict.copy()
            offer_input["predicted_price"] = sell_price

            # 모든 시나리오 결과 계산
            all_offers = {}
            for sc_name in SCENARIOS.keys():
                all_offers[sc_name] = calculate_purchase_offer(offer_input, sc_name, repair_cost=mac_repair_cost)

            # 시나리오별 요약 테이블
            summary_df = pd.DataFrame({
                '시나리오': list(all_offers.keys()),
                '예상 판매가': [f"{o['target_sale_price']:,}원" for o in all_offers.values()],
                '최종 매입가': [f"{o['final_purchase_price']:,}원" for o in all_offers.values()],
                '판매 프리미엄': [f"{o['sale_premium_rate']*100:.1f}%" for o in all_offers.values()],
                '운영비율': [f"{o['op_cost_rate']*100:.1f}%" for o in all_offers.values()],
                '목표 마진율': [f"{o['target_margin_rate']*100:.1f}%" for o in all_offers.values()],
                '안전 마진': [f"{o['safety_margin_rate']*100:.1f}%" for o in all_offers.values()],
                '총 차감률': [f"{o['total_deduction_rate']*100:.1f}%" for o in all_offers.values()],
                '예상 스프레드': [f"{o['expected_gross_spread']:,}원" for o in all_offers.values()],
                '매입/예측 비율': [f"{o['implied_purchase_ratio_vs_pred']*100:.1f}%" for o in all_offers.values()],
            })
            st.table(summary_df)

            # 표준 시나리오 메트릭 강조
            std_offer = all_offers["표준"]
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            with col_m1:
                st.metric("예측가 (기준)", f"{sell_price:,.0f}원", help=sell_label)
            with col_m2:
                st.metric("권장 매입가", f"{std_offer['final_purchase_price']:,}원")
            with col_m3:
                st.metric("수리비용", f"{std_offer['repair_cost']:,}원")
            with col_m4:
                st.metric("권장 판매가", f"{std_offer['target_sale_price']:,}원")
            with col_m5:
                st.metric("마진", f"{std_offer['expected_gross_spread']:,}원")

            # 재고 기간별 시뮬레이션
            st.markdown("📦 **재고 기간별 가격 하락 시뮬레이션**")
            st.caption("위플레닛 실제 데이터: 평균 재고 359일, 50.4%가 1년 이상 재고")
            판매가_즉시 = std_offer["target_sale_price"]
            판매가_보통 = int(판매가_즉시 * 0.95)
            판매가_장기 = int(판매가_즉시 * 0.90)
            매입가 = std_offer["final_purchase_price"]
            마진_즉시 = 판매가_즉시 - 매입가
            마진_보통 = 판매가_보통 - 매입가
            마진_장기 = 판매가_장기 - 매입가
            마진율_즉시 = (마진_즉시 / 판매가_즉시 * 100) if 판매가_즉시 > 0 else 0
            마진율_보통 = (마진_보통 / 판매가_보통 * 100) if 판매가_보통 > 0 else 0
            마진율_장기 = (마진_장기 / 판매가_장기 * 100) if 판매가_장기 > 0 else 0

            inv_df = pd.DataFrame({
                '기간': ['빠른 판매 (~6개월, 25.4%)', '보통 (6개월~1년, 24.1%, -5%)', '장기 재고 (1년+, 50.4%, -10%)'],
                '판매가': [f"{판매가_즉시:,}원", f"{판매가_보통:,}원", f"{판매가_장기:,}원"],
                '마진': [f"{마진_즉시:,}원", f"{마진_보통:,}원", f"{마진_장기:,}원"],
                '마진율': [f"{마진율_즉시:.1f}%", f"{마진율_보통:.1f}%", f"{마진율_장기:.1f}%"]
            })
            st.table(inv_df)

        # 모델 성능
        st.markdown("---")
        st.subheader("🎯 모델 성능")
        col_r, col_m = st.columns(2)
        with col_r:
            st.metric("R² (RF)", f"{mac_rf_r2:.4f}")
        with col_m:
            st.metric("MAE (RF)", f"{mac_rf_mae:,.0f}원")


# ===== 아이폰 =====
if device == '아이폰':
    st.markdown("""
    <div style='background: linear-gradient(90deg,#E3F2FD,#BBDEFB); border-left:5px solid #1565C0;
    border-radius:10px; padding:14px 20px; margin-bottom:16px;'>
    <span style='font-size:1.15rem; font-weight:700; color:#0D47A1;'>📱 아이폰 매입가 자동 산정 시스템</span><br>
    <span style='font-size:0.97rem; color:#1565C0;'>
    실거래 데이터 기반 <b>XGBoost PriceNet</b> (R²=0.9076, MAE≈94,373원) ·
    <b>전략별 마진 시뮬레이터</b> 통합 탑재 — 모델·용량·상태 입력만으로 매입 적정가를 즉시 산출합니다.
    </span>
    </div>
    """, unsafe_allow_html=True)

    # ===== 아이폰 통합 모델: XGBoost PriceNet (pkl 로드) =====
    @st.cache_resource
    def load_iphone_models():
        import joblib
        d = joblib.load('models/iphone_model.pkl')
        return d['model'], d['features'], d['df_r'], d['r2'], d['mae'], d['mape']

    xgb_model, iphone_features, df_r_iphone, iphone_r2, iphone_mae, iphone_mape = load_iphone_models()

    # ===== 사이드바 =====
    with st.sidebar:
        st.markdown("---")
        st.header("📱 아이폰 스펙 입력")
        # 출시일 내림차순 정렬된 df_r 기준으로 중복 제거 (순서 유지)
        iphone_model_list = list(dict.fromkeys(df_r_iphone['모델명'].dropna())) if df_r_iphone is not None else []
        sel_model = st.selectbox("모델명", iphone_model_list, key='ip_model')
        # 용량도 해당 모델 내에서 출시일 순서 유지
        storage_options = list(dict.fromkeys(
            df_r_iphone[df_r_iphone['모델명'] == sel_model]['용량'].dropna()
        )) if df_r_iphone is not None else ['128GB']
        sel_storage = st.selectbox("용량", storage_options, key='ip_storage')

        st.divider()
        ip_cond_type = st.radio("상품 종류", ["중고 기기", "완전 미개봉", "단순 개봉"], key='ip_cond')
        in_brand_new = 1 if ip_cond_type == "완전 미개봉" else 0
        in_simple_open = 1 if ip_cond_type == "단순 개봉" else 0

        st.divider()
        show_iphone_detail = st.checkbox("상태 상세 입력", key='iphone_detail')
        if show_iphone_detail:
            st.subheader("개별 결함 선택")
            in_crack = st.checkbox("액정 파손/금", key='ip_crack')
            in_burn = st.checkbox("번인/화면 잔상", key='ip_burn')
            in_dent = st.checkbox("테두리 찍힘/까짐", key='ip_dent')
            in_scratch = st.checkbox("생활 기스/스크래치", key='ip_scratch')
            st.divider()
            iphone_battery = st.slider("배터리 효율 (%)", 70, 100, 90, key='iphone_battery')
            in_unoff = st.checkbox("사설 수리 이력", key='ip_unoff')
            in_apc = st.checkbox("애플케어플러스", key='ip_apc')
        else:
            in_crack = False; in_burn = False; in_dent = False; in_scratch = False
            iphone_battery = 90; in_unoff = False; in_apc = False

        st.divider()
        show_iphone_margin = st.checkbox("마진 시뮬레이션", key='iphone_margin')

    # ===== 출시 정보 조회 =====
    if df_r_iphone is None or xgb_model is None:
        st.error("아이폰 데이터 로딩 실패. 파일 경로를 확인해주세요.")
    else:
        ref_rows = df_r_iphone[(df_r_iphone['모델명'] == sel_model) & (df_r_iphone['용량'] == sel_storage)]
        if ref_rows.empty:
            st.warning("해당 모델/용량 조합의 출시가 정보가 없습니다.")
        else:
            ref = ref_rows.iloc[0]
            launch_price_raw = str(ref['공식출시가(원)']).replace(',', '').replace(' ', '')
            launch_price = int(re.sub(r'[^0-9]', '', launch_price_raw)) if re.sub(r'[^0-9]', '', launch_price_raw) else 0
            launch_date = pd.to_datetime(ref['한국 출시일'], errors='coerce')
            months_old_now = max(((datetime(2026, 3, 20) - launch_date).days // 30), 0) if pd.notna(launch_date) else 24
            storage_num = float(re.search(r'(\d+)', sel_storage).group(1)) if re.search(r'(\d+)', sel_storage) else 128.0
            is_high_end = 1 if re.search(r'Pro|Max', sel_model, re.I) else 0

            # usage_intensity를 현재 시점으로 고정 (미래 예측 시 가격 역전 방지)
            current_intensity = (100 - iphone_battery) / (months_old_now + 1)

            def predict_iphone(m_offset=0, bat=90, crack=0, burn=0, dent=0, scratch=0, unoff=0, apc=0, brand_new=0, simple_open=0):
                f_m = months_old_now + m_offset
                inp = pd.DataFrame([[
                    float(launch_price), float(f_m), storage_num, is_high_end, current_intensity,
                    brand_new, simple_open, crack, burn, dent, scratch, unoff, apc
                ]], columns=iphone_features)
                pred_val = int(np.expm1(xgb_model.predict(inp)[0]))
                # 안전장치: 미래 가격이 현재 가격보다 높을 수 없음
                if m_offset > 0:
                    return min(pred_val, predict_iphone(0, bat, crack, burn, dent, scratch, unoff, apc, brand_new, simple_open))
                return pred_val

            current_p = predict_iphone(
                bat=iphone_battery,
                crack=int(in_crack), burn=int(in_burn), dent=int(in_dent), scratch=int(in_scratch),
                unoff=int(in_unoff), apc=int(in_apc),
                brand_new=in_brand_new, simple_open=in_simple_open
            )
            dep_residual = (current_p / launch_price * 100) if launch_price > 0 else 0

            # === 섹션 1: 현재시세 예측 ===
            st.subheader("💰 1. 현재시세 예측")
            st.caption("XGBoost PriceNet — 모델·용량·개봉여부·결함·배터리·사설수리·AppleCare 13개 변수 기반 시장가 예측")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("예상 시장가", f"{current_p:,.0f}원")
            with col_b:
                st.metric("공식 출시가", f"{launch_price:,.0f}원")
            with col_c:
                st.metric("잔존가치율", f"{dep_residual:.1f}%  (▼{100-dep_residual:.1f}%)")

            months_passed_display = months_old_now
            st.info(
                f"**{sel_model} {sel_storage}** | 출시가 {launch_price:,}원 | "
                f"경과 {months_passed_display}개월 | 배터리 {iphone_battery}% | "
                f"**예상 시장가 {current_p:,.0f}원** (잔존가치 {dep_residual:.1f}%)"
            )
            st.caption(
                f"XGBoost PriceNet | R²: {iphone_r2:.4f} | MAE: {iphone_mae:,.0f}원 | MAPE: {iphone_mape:.2f}%"
            )

            # === 섹션 2: 마진 시뮬레이션 ===
            if show_iphone_margin:
                st.markdown("---")
                st.subheader("📈 2. 매입 마진 시뮬레이션")
                st.caption("전략별 추천 매입가 및 실질 순이익 산출")

                # 전략 선택 및 추천 매입가 계산
                ip_strategy = st.radio("목표 전략 선택", ["공격적", "적정", "보수적"], horizontal=True, index=1, key='ip_strategy')
                margin_map = {"공격적": 0.10, "적정": 0.16, "보수적": 0.23}
                f_margin = margin_map[ip_strategy] + (iphone_mape / 200)
                suggested_buy = int(current_p * (1 - f_margin))

                c1, c2 = st.columns(2)
                c1.metric("현재 예측 시세", f"{current_p:,.0f}원")
                c2.metric("추천 매입가", f"{suggested_buy:,.0f}원", delta=f"목표마진 {f_margin*100:.1f}%")

                st.divider()

                # 비용 수기 입력
                st.write("**💰 비용 수기 입력**")
                c_cost1, c_cost2, c_cost3 = st.columns(3)
                with c_cost1:
                    input_repair_cost = st.number_input("수리비 (원)", value=0, step=1000, key='ip_repair')
                with c_cost2:
                    input_operating_cost = st.number_input("기타 운영비 (원)", value=0, step=1000, key='ip_opex')
                with c_cost3:
                    shipping_cost = 5000
                    st.write("🚚 고정 배송비")
                    st.write(f"**{shipping_cost:,}원**")

                total_additional_cost = input_repair_cost + input_operating_cost + shipping_cost
                net_profit = (current_p - suggested_buy) - total_additional_cost
                st.write(f"#### 현재 기준 예상 순이익: :blue[{net_profit:,.0f}원]")

                st.divider()

                # 재고 보유 기간별 감가 시뮬레이션
                st.subheader("📉 재고 보유 기간별 감가 시뮬레이션")
                periods = [0, 1, 3, 6, 12]
                period_labels = ["즉시 판매", "1개월 뒤", "3개월 뒤", "6개월 뒤", "12개월 뒤"]
                dep_data = []
                for m, lbl in zip(periods, period_labels):
                    p_val = predict_iphone(
                        m_offset=m, bat=iphone_battery,
                        crack=int(in_crack), burn=int(in_burn), dent=int(in_dent), scratch=int(in_scratch),
                        unoff=int(in_unoff), apc=int(in_apc),
                        brand_new=in_brand_new, simple_open=in_simple_open
                    )
                    current_net = (p_val - suggested_buy) - total_additional_cost
                    dep_data.append({"기간": lbl, "예상 시세": p_val, "마진": current_net})

                dep_margin_df = pd.DataFrame(dep_data)
                dep_margin_df['기간'] = pd.Categorical(dep_margin_df['기간'], categories=period_labels, ordered=True)
                dep_margin_df = dep_margin_df.sort_values('기간')

                col_chart2, col_table2 = st.columns([2, 1])
                with col_chart2:
                    st.line_chart(dep_margin_df.set_index("기간")["예상 시세"])
                with col_table2:
                    st.write("📋 **상세 데이터**")
                    st.dataframe(dep_margin_df.set_index("기간"), use_container_width=True)

            st.divider()
            st.info(
                f"**📊 시스템 분석 정보:** "
                f"XGBoost PriceNet | R² `{iphone_r2:.4f}` | MAE `{iphone_mae:,.0f}원` | "
                f"MAPE `{iphone_mape:.2f}%` (예측 불확실성 대비 안전 마진 포함)"
            )
