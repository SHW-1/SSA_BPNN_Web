import os
import numpy as np
import streamlit as st
from scipy.io import loadmat


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="SSA-BPNN Drying Shrinkage Prediction",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# Model file path
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "SSA_BPNN_Web_Model.mat")


# =========================================================
# Scale setting
# =========================================================
# 你的 DryShrinkage 是 4.65 这种量级，所以这里保持 1.0
OUTPUT_SCALE_OVERRIDE = 1.0

# 残差区间也保持同一量级
RESIDUAL_SCALE_OVERRIDE = 1.0


# =========================================================
# Input features
# 顺序必须和 MATLAB 训练时输入顺序完全一致：
# Cement, FlyAsh, CoarseAgg, SandRatio, W_B, RCA_Ratio, Absorption, Time
# =========================================================
BASE_FEATURES = [
    {
        "key": "cement",
        "name": "Cement",
        "unit": "kg/m³"
    },
    {
        "key": "flyash",
        "name": "Fly ash",
        "unit": "kg/m³"
    },
    {
        "key": "coarseagg",
        "name": "Coarse aggregate",
        "unit": "kg/m³"
    },
    {
        "key": "sandratio",
        "name": "Sand ratio",
        "unit": ""
    },
    {
        "key": "wb",
        "name": "Water-binder ratio",
        "unit": ""
    },
    {
        "key": "rca_ratio",
        "name": "RCA replacement ratio",
        "unit": "%"
    },
    {
        "key": "absorption",
        "name": "Mixed coarse aggregate water absorption",
        "unit": "%"
    },
    {
        "key": "time",
        "name": "Drying shrinkage test age",
        "unit": "d"
    }
]


# =========================================================
# Helper functions
# =========================================================
def get_required(mat_data, name):
    if name not in mat_data:
        raise KeyError(f"Missing variable in .mat file: {name}")
    return mat_data[name]


def has_var(mat_data, name):
    return name in mat_data


def to_column(x):
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 0:
        return arr.reshape(1, 1)

    return arr.reshape(-1, 1)


def to_1d(x):
    return np.asarray(x, dtype=float).reshape(-1)


def to_scalar(x):
    arr = np.asarray(x, dtype=float).reshape(-1)

    if arr.size == 0:
        raise ValueError("Empty scalar value.")

    return float(arr[0])


def to_matrix(x, name="matrix"):
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)

    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)

    elif arr.ndim == 2:
        pass

    else:
        raise ValueError(f"{name} has unsupported shape: {arr.shape}")

    return arr


def matlab_string(x, default_value):
    if x is None:
        return default_value

    if isinstance(x, str):
        return x.strip()

    arr = np.asarray(x)

    if arr.dtype.kind in ["U", "S"]:
        return "".join(arr.reshape(-1).astype(str).tolist()).strip()

    try:
        return str(arr.item()).strip()
    except Exception:
        return default_value


def safe_vector_from_mat(mat_data, name, default=None):
    if name not in mat_data:
        return default

    try:
        return to_1d(mat_data[name])
    except Exception:
        return default


# =========================================================
# SSA-BPNN model class
# =========================================================
class SSABPNNModel:
    def __init__(self, model_file):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        mat_data = loadmat(
            model_file,
            squeeze_me=True,
            struct_as_record=False
        )

        # ========== 1. Network parameters ==========
        self.IW = to_matrix(get_required(mat_data, "IW"), "IW")
        self.LW = to_matrix(get_required(mat_data, "LW"), "LW")

        self.b1 = to_column(get_required(mat_data, "b1"))
        self.b2 = to_column(get_required(mat_data, "b2"))

        # ========== 2. Activation functions ==========
        if has_var(mat_data, "hidden_transferFcn"):
            self.hidden_transfer = matlab_string(mat_data["hidden_transferFcn"], "tansig")
        else:
            self.hidden_transfer = "tansig"

        if has_var(mat_data, "output_transferFcn"):
            self.output_transfer = matlab_string(mat_data["output_transferFcn"], "purelin")
        else:
            self.output_transfer = "purelin"

        # ========== 3. External mapminmax parameters ==========
        self.input_xoffset = to_column(get_required(mat_data, "input_xoffset"))
        self.input_gain = to_column(get_required(mat_data, "input_gain"))
        self.input_ymin = to_column(get_required(mat_data, "input_ymin"))

        self.output_xoffset = to_column(get_required(mat_data, "output_xoffset"))
        self.output_gain = to_column(get_required(mat_data, "output_gain"))
        self.output_ymin = to_column(get_required(mat_data, "output_ymin"))

        # ========== 4. Output scale ==========
        if has_var(mat_data, "output_scale"):
            self.output_scale_from_mat = to_scalar(mat_data["output_scale"])
        else:
            self.output_scale_from_mat = 1.0

        if OUTPUT_SCALE_OVERRIDE is None:
            self.output_scale = self.output_scale_from_mat
        else:
            self.output_scale = float(OUTPUT_SCALE_OVERRIDE)

        if self.output_scale == 0:
            raise ValueError("output_scale cannot be zero.")

        # ========== 5. MATLAB internal process matrices ==========
        self.has_internal_process = (
            has_var(mat_data, "input_process_A")
            and has_var(mat_data, "input_process_b")
            and has_var(mat_data, "output_reverse_A")
            and has_var(mat_data, "output_reverse_b")
        )

        if self.has_internal_process:
            self.input_process_A = to_matrix(mat_data["input_process_A"], "input_process_A")
            self.input_process_b = to_column(mat_data["input_process_b"])

            self.output_reverse_A = to_matrix(mat_data["output_reverse_A"], "output_reverse_A")
            self.output_reverse_b = to_column(mat_data["output_reverse_b"])
        else:
            self.input_process_A = np.eye(8)
            self.input_process_b = np.zeros((8, 1))

            self.output_reverse_A = np.eye(1)
            self.output_reverse_b = np.zeros((1, 1))

        # ========== 6. Feature default/min/max ==========
        self.feature_default = safe_vector_from_mat(mat_data, "feature_default", None)
        self.feature_min = safe_vector_from_mat(mat_data, "feature_min", None)
        self.feature_max = safe_vector_from_mat(mat_data, "feature_max", None)

        if self.feature_default is None or len(self.feature_default) != 8:
            self.feature_default = np.array(
                [300, 0, 1154, 0.4, 0.536, 0, 1.7, 1],
                dtype=float
            )

        # ========== 7. Residual interval ==========
        self.has_residual_interval = False
        self.residual_low95 = None
        self.residual_high95 = None

        if has_var(mat_data, "residual_low95") and has_var(mat_data, "residual_high95"):
            self.residual_low95 = to_scalar(mat_data["residual_low95"]) / RESIDUAL_SCALE_OVERRIDE
            self.residual_high95 = to_scalar(mat_data["residual_high95"]) / RESIDUAL_SCALE_OVERRIDE
            self.has_residual_interval = True

        self.check_dimensions()

    def check_dimensions(self):
        if self.input_xoffset.shape[0] != 8:
            raise ValueError(
                f"input_xoffset should contain 8 values, got shape {self.input_xoffset.shape}"
            )

        if self.input_gain.shape[0] != 8:
            raise ValueError(
                f"input_gain should contain 8 values, got shape {self.input_gain.shape}"
            )

        if self.input_process_A.shape[1] != 8:
            raise ValueError(
                f"input_process_A should have 8 columns, got shape {self.input_process_A.shape}"
            )

        processed_input_size = self.input_process_A.shape[0]

        if self.IW.shape[1] != processed_input_size:
            if self.IW.shape[0] == processed_input_size:
                self.IW = self.IW.T
            else:
                raise ValueError(
                    f"IW shape does not match input_process_A. "
                    f"IW shape: {self.IW.shape}, input_process_A shape: {self.input_process_A.shape}"
                )

        hidden_size = self.IW.shape[0]

        if self.b1.shape[0] != hidden_size:
            raise ValueError(
                f"b1 size does not match hidden layer size. "
                f"b1 shape: {self.b1.shape}, hidden size: {hidden_size}"
            )

        if self.LW.shape[1] != hidden_size:
            if self.LW.shape[0] == hidden_size and self.LW.shape[1] == 1:
                self.LW = self.LW.T
            else:
                raise ValueError(
                    f"LW shape does not match hidden layer size. "
                    f"LW shape: {self.LW.shape}, hidden size: {hidden_size}"
                )

        output_size = self.LW.shape[0]

        if self.b2.shape[0] != output_size:
            raise ValueError(
                f"b2 size does not match output size. "
                f"b2 shape: {self.b2.shape}, output size: {output_size}"
            )

        if self.output_reverse_A.shape[1] != output_size:
            raise ValueError(
                f"output_reverse_A does not match output size. "
                f"output_reverse_A shape: {self.output_reverse_A.shape}, output size: {output_size}"
            )

    @staticmethod
    def tansig(x):
        return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

    @staticmethod
    def logsig(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def purelin(x):
        return x

    def apply_activation(self, x, name):
        name = str(name).lower().strip()

        if name == "tansig":
            return self.tansig(x)

        if name == "logsig":
            return self.logsig(x)

        if name == "purelin":
            return self.purelin(x)

        raise ValueError(f"Unsupported activation function: {name}")

    def normalize_input(self, x):
        return (x - self.input_xoffset) * self.input_gain + self.input_ymin

    def reverse_output(self, y_norm):
        return (y_norm - self.output_ymin) / self.output_gain + self.output_xoffset

    def predict(self, input_values):
        x = np.asarray(input_values, dtype=float).reshape(-1, 1)

        if x.shape[0] != 8:
            raise ValueError("The model requires exactly 8 input parameters.")

        # 1. External input normalization
        x_norm = self.normalize_input(x)

        # 2. MATLAB network internal input processing
        x_net = self.input_process_A @ x_norm + self.input_process_b

        # 3. Hidden layer
        hidden_input = self.IW @ x_net + self.b1
        hidden_output = self.apply_activation(hidden_input, self.hidden_transfer)

        # 4. Output layer
        output_input = self.LW @ hidden_output + self.b2
        output_internal = self.apply_activation(output_input, self.output_transfer)

        # 5. MATLAB network internal output reverse processing
        y_norm = self.output_reverse_A @ output_internal + self.output_reverse_b

        # 6. External reverse mapminmax
        y_raw = self.reverse_output(y_norm)

        # 7. Output scale conversion
        prediction = float(y_raw.reshape(-1)[0]) / self.output_scale

        # 8. 95% predictive interval
        if self.has_residual_interval:
            raw_low = prediction + self.residual_low95
            raw_high = prediction + self.residual_high95

            # 预测区间下限最小值限制为 0
            low_bound = max(0.0, raw_low)

            # 上限也不允许小于 0，同时保证上限不小于下限
            high_bound = max(0.0, raw_high)

            if high_bound < low_bound:
                high_bound = low_bound

            interval_method = "test residual calibration"
        else:
            low_bound = None
            high_bound = None
            interval_method = "unavailable"

        return prediction, low_bound, high_bound, interval_method


# =========================================================
# Load model with cache
# =========================================================
@st.cache_resource
def load_model():
    return SSABPNNModel(MODEL_FILE)


try:
    model = load_model()
except Exception as e:
    st.error("Model loading failed. Please check SSA_BPNN_Web_Model.mat.")
    st.exception(e)
    st.stop()


# =========================================================
# Custom page style
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f2f2f2;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        text-align: center;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #111827;
        margin-bottom: 42px;
    }

    .section-title {
        font-size: 15px;
        font-weight: 600;
        color: #111111;
        margin-bottom: 8px;
    }

    div[data-testid="stNumberInput"] {
        border: 1px solid #8a8a8a;
        padding: 4px 8px 10px 8px;
        background-color: #f8f8f8;
    }

    div[data-testid="stNumberInput"] label {
        font-size: 13px;
        color: #111111;
        font-weight: 400;
    }

    div[data-testid="stNumberInput"] input {
        text-align: center;
        background-color: white;
        color: black;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        height: 64px;
        border: 1px solid #8a8a8a;
        border-radius: 5px;
        background-color: white;
        color: black;
        font-size: 16px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    div[data-testid="stButton"] > button:hover {
        border: 1px solid #555555;
        color: black;
        background-color: #fafafa;
    }

    .result-box {
        background-color: #eaffef;
        padding: 26px 20px;
        text-align: center;
        font-size: 17px;
        font-weight: 700;
        color: #006b2d;
        margin-top: 4px;
        margin-bottom: 20px;
    }

    .interval-box {
        background-color: #f3f4fb;
        padding: 24px 20px;
        text-align: center;
        font-size: 15px;
        font-weight: 700;
        color: #111827;
        line-height: 1.5;
    }

    .footer-text {
        text-align: center;
        color: #777777;
        font-size: 12px;
        margin-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Page title
# =========================================================
st.markdown(
    """
    <div class="main-title">
        Prediction of RAC Drying Shrinkage Based on SSA-BPNN
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Input section
# =========================================================
with st.container(border=True):
    st.markdown('<div class="section-title">Input Parameters</div>', unsafe_allow_html=True)

    input_values = []

    display_features = [
        {
            "key": "cement",
            "label": "Ordinary Portland cement content (kg/m³)",
            "step": 1.0
        },
        {
            "key": "flyash",
            "label": "Fly ash content (kg/m³)",
            "step": 1.0
        },
        {
            "key": "coarseagg",
            "label": "Coarse aggregate content (kg/m³)",
            "step": 1.0
        },
        {
            "key": "sandratio",
            "label": "Sand ratio",
            "step": 0.001
        },
        {
            "key": "wb",
            "label": "Water-binder ratio",
            "step": 0.001
        },
        {
            "key": "rca_ratio",
            "label": "Recycled coarse aggregate replacement ratio (%)",
            "step": 1.0
        },
        {
            "key": "absorption",
            "label": "Mixed coarse aggregate water absorption (%)",
            "step": 0.1
        },
        {
            "key": "time",
            "label": "Drying shrinkage test age (d)",
            "step": 1.0
        }
    ]

    for row in range(4):
        cols = st.columns(2)

        for col_idx in range(2):
            i = row * 2 + col_idx
            feature = display_features[i]

            with cols[col_idx]:
                default_value = float(model.feature_default[i])

                value = st.number_input(
                    feature["label"],
                    value=default_value,
                    step=feature["step"],
                    format="%.6f",
                    key=feature["key"]
                )

                input_values.append(value)


# =========================================================
# Prediction calculation
# =========================================================
if "prediction_value" not in st.session_state:
    st.session_state.prediction_value = None

if "prediction_low" not in st.session_state:
    st.session_state.prediction_low = None

if "prediction_high" not in st.session_state:
    st.session_state.prediction_high = None

if st.button("Predict"):
    try:
        prediction, low_bound, high_bound, interval_method = model.predict(input_values)

        st.session_state.prediction_value = prediction
        st.session_state.prediction_low = low_bound
        st.session_state.prediction_high = high_bound

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)


# =========================================================
# Result display
# =========================================================
if st.session_state.prediction_value is None:
    result_text = "Drying shrinkage prediction: --"
else:
    result_text = f"Drying shrinkage prediction: {st.session_state.prediction_value:.6f}"

st.markdown(
    f"""
    <div class="result-box">
        {result_text}
    </div>
    """,
    unsafe_allow_html=True
)


if (
    st.session_state.prediction_low is None
    or st.session_state.prediction_high is None
):
    interval_text = """
        Uncertainty Analysis (95% Predictive Interval)<br>
        Prediction interval: --
    """
else:
    interval_text = f"""
        Uncertainty Analysis (95% Predictive Interval)<br>
        Prediction interval: [{st.session_state.prediction_low:.6f}, {st.session_state.prediction_high:.6f}]
    """

st.markdown(
    f"""
    <div class="interval-box">
        {interval_text}
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Footer
# =========================================================
st.markdown(
    """
    <div class="footer-text">
        SSA-BPNN prediction system for recycled aggregate concrete drying shrinkage
    </div>
    """,
    unsafe_allow_html=True
)