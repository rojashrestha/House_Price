import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# PAGE CONFIGURATION: Web App ko basic settings ra appearance set gareko.
# IMPORTANT RULE: Yo function Streamlit app ko FIRST line code (imports paxi) 
# hunuparchha, natra Streamlit le error dinchha.
# ==============================================================================
st.set_page_config(
    # 1. Browser ko tab/window ma dekhine naya title set gareko (Chrome Tab Name)
    page_title="House Price Predictor",
    
    # 2. Browser tab ma title ko agadi dekhine icon (emoji or image favicon)
    page_icon="🏠",
    
    # 3. Layout wide banako: web page ko screen size maximum stretch garchha.
    #    Default centering (sano) ko satta "wide" (thulo screen) use gareko plots ra columns ko lagi.
    layout="wide",  
    
    # 4. Sidebar ko starting view configuration: Page load hune bitikai 
    #    Side navigation panel open status ("expanded") ma basos bhannako lagi.
    initial_sidebar_state="expanded"
)


# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #2e4053;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 16px;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 40px;
    }
            
            /* 1. Slider ko tracking line (active bar) color change */
    div[data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #6c5ce7 0%, #a29bfe 100%) !important;
    }
    
    /* 2. Slider ko drag handle button color change */
    div[role="slider"] {
        background-color: #pink important;
        border: 2px solid #a29bfe !important;
        box-shadow: 0 0 8px rgba(108, 92, 231, 0.4) !important;
    }
            /* Each tab button ko bich ma gap (space) thapna ko lagi */
button[data-baseweb="tab"] {
    margin-right: 410px !important; /* 40px ko gap dinchha (sarna / control garna sakincha) */
    padding-left: 20px !important;  /* Tab bhitra left padding */
    padding-right: 20px !important; /* Tab bhitra right padding */
}
    </style>
""", unsafe_allow_html=True)

# # ============================================
# # 2. LINEAR REGRESSION CLASS (Scratch Model)
# # ============================================
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def compute_cost(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []
        
        for _ in range(self.epochs):
            y_pred = self.predict(X)
            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


# ============================================
# 3. CACHED DATA GENERATION & UTILITIES
# ============================================
# @st.cache_data le garda user le page click garda data baranbar change (regenerate) hudaina
@st.cache_data
def generate_dataset(n=200):
    np.random.seed(42)
    area = np.random.randint(500, 4000, size=n)
    bedrooms = np.random.randint(1, 7, size=n)
    age = np.random.randint(0, 51, size=n)
    noise = np.random.normal(0, 15000, size=n)
    
    price = (area * 120) + (bedrooms * 8000) - (age * 1500) + noise
    
    df = pd.DataFrame({
        'Area': area,
        'Bedrooms': bedrooms,
        'Age': age,
        'Price': price
    })
    return df

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

def train_test_split(X, y, test_size=0.2):
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]


# # ============================================
# # 4. APP FLOW & STATE MANAGEMENT
# # ============================================
# Title Displays
st.markdown('<div class="main-title"> House Price Predictor</div>', unsafe_allow_html=True)


# Generate Dataset
df = generate_dataset()
X = df[['Area', 'Bedrooms', 'Age']].values
y = df['Price'].values

# Sidebar controls (Hyperparameters)
st.sidebar.header(" Model Settings")
lr = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, step=0.001)
epochs = st.sidebar.slider("Epochs (Iterations)", 100, 5000, 1000, step=100)
test_size = st.sidebar.slider("Test Size (Split ratio)", 0.1, 0.4, 0.2, step=0.05)

# Normalization & Split
X_norm, X_mean, X_std = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=test_size)

# Cache model training so it doesn't retrain on every single slider movement
@st.cache_resource
def train_model(X_tr, y_tr, learning_rate, num_epochs):
    model = LinearRegressionScratch(learning_rate=learning_rate, epochs=num_epochs)
    model.fit(X_tr, y_tr)
    return model

# Train model
model = train_model(X_train, y_train, lr, epochs)

# # ============================================
# # 5. TABS LAYOUT (UI/UX Design)
# # ============================================
# 
tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🧠 Model Metrics & Training", "🔮 Make Predictions"])


# --- TAB 1: DATA EXPLORER ---
with tab1:
    st.header("📊 Dataset Overview")
    st.write("Below is the generated synthetic house price dataset used for training the model.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Raw Data (Sample)")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.subheader("Quick Statistics")
        st.write(df.describe())

# --- TAB 2: MODEL TRAINING ---
with tab2:
    st.header("🧠 Model Metrics")
    
    # Predict and calculate accuracy metrics
    y_pred = model.predict(X_test)
    mse = model.compute_cost(y_test, y_pred)
    
    # Custom R2 Score calculation
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
# ============================================
# ROW 1: Metrics (2 columns)
# ============================================
m_col1, m_col2 = st.columns(2)
m_col1.metric(label="Mean Squared Error (MSE)", value=f"${mse:,.2f}")
m_col2.metric(label="R² Accuracy Score", value=f"{r2:.4f}")

st.markdown("---") # Divider line 

# ============================================
# ROW 2: Visualizations (2 columns)
# ============================================
p_col1, p_col2 = st.columns(2)

with p_col1:
    st.subheader("Training Loss Curve")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(model.cost_history, color="#bf0909", linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with p_col2:
    st.subheader("Actual vs Predicted")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(y_test, y_pred, color='#3498db', alpha=0.6, edgecolors='navy')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('Actual Price')
    ax2.set_ylabel('Predicted Price')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
   


# --- TAB 3: MAKE PREDICTIONS ---
with tab3:
    st.header("🔮 Input House Details")
    st.write("Adjust the parameters below to predict the price of a house using the trained weights.")
    
    col1, col2, col3 = st.columns(3)
    user_area = col1.number_input("Area (in Sq Ft)", min_value=500, max_value=5000, value=1500, step=50)
    user_bedrooms = col2.number_input("Number of Bedrooms", min_value=1, max_value=6, value=3, step=1)
    user_age = col3.number_input("Age of the House (Years)", min_value=0, max_value=50, value=10, step=1)
    
    if st.button("💰 Predict House Price", use_container_width=True):
        # 1. Package input data
        user_data = np.array([[user_area, user_bedrooms, user_age]])
        
        # 2. Normalize input using training mean & std!
        user_data_norm = (user_data - X_mean) / X_std
        
        # 3. Model prediction
        predicted_price = model.predict(user_data_norm)[0]
        
        # 4. Display result with style
        st.success(f"### predicted price: ${predicted_price:,.2f}")
        
        # Show formula used dynamically
        st.info(f"**How it works (Learned Weight Calculation):**\n"
                f"Price = ({user_area} × {model.weights[0]:.2f}) + "
                f"({user_bedrooms} × {model.weights[1]:.2f}) + "
                f"({user_age} × {model.weights[2]:.2f}) + "
                f"Bias ({model.bias:,.2f})")