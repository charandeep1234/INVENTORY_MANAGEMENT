import os 
import sqlite3 
from datetime import datetime 
 
import numpy as np 
import pandas as pd 
import plotly.express as px 
import streamlit as st 
import torch 
import torch.nn as nn 
import torch.optim as optim 
 
 
# ============================================================ 
# PAGE CONFIG 
# ============================================================ 
st.set_page_config( 
    page_title="AI Inventory Optimization", 
    page_icon="📦", 
    layout="wide", 
    initial_sidebar_state="expanded", 
) 
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DB_PATH = os.path.join(BASE_DIR, "inventory.db") 
MODEL_DIR = os.path.join(BASE_DIR, "models") 
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_model.pth") 
 
os.makedirs(MODEL_DIR, exist_ok=True) 
 
REQUIRED_COLUMNS = [ 
    "Date", 
    "Product_ID", 
    "Demand", 
    "Inventory_Level", 
    "Lead_Time", 
    "Unit_Cost", 
    "Holding_Cost", 
    "Stockout_Cost", 
    "Ordering_Cost", 
] 
 
ACTIONS = np.array([0, 20, 40, 60, 80, 100], dtype=np.float32) 
 
 
# ============================================================ 
# PROFESSIONAL UI 
# ============================================================ 
st.markdown( 
    """ 
    <style> 
    :root { 
        --navy: #123b5d; 
        --navy-deep: #0d2b45; 
        --blue: #1f6f8b; 
        --teal: #159a9c; 
        --page: #f4f7fa; 
        --card: #ffffff; 
        --text: #17324d; 
        --muted: #526777; 
        --border: #d6e0e8; 
        --success: #16855b; 
        --warning: #9a6500; 
        --danger: #c0392b; 
    } 
 
    .stApp { 
        background: 
        radial-gradient(circle at 10% 10%, rgba(21,154,156,.09), transparent 25%), 
        radial-gradient(circle at 90% 20%, rgba(31,111,139,.08), transparent 25%), 
        linear-gradient(135deg, #f8fbfd 0%, var(--page) 52%, #edf4f7 100%); 
        color: var(--text); 
    } 
 
    .block-container { 
        max-width: 1450px; 
        padding-top: 2rem; 
        padding-bottom: 3rem; 
    } 
 
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, var(--navy-deep) 0%, var(--navy) 100%); 
    } 
 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"], 
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { 
        color: #ffffff !important; 
    } 
 
    [data-testid="stSidebar"] [role="radiogroup"] { 
        gap: 5px; 
    } 
 
    [data-testid="stSidebar"] [role="radiogroup"] label { 
        background: rgba(255,255,255,.06); 
        border: 1px solid rgba(255,255,255,.12); 
        border-radius: 8px; 
        padding: 9px 10px; 
        transition: background .18s ease, border-color .18s ease; 
    } 
 
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { 
        background: rgba(21,154,156,.25); 
        border-color: rgba(159,238,229,.7); 
    } 
 
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { 
        background: var(--teal); 
        border-color: #8ce3dc; 
        box-shadow: 0 3px 10px rgba(0,0,0,.18); 
    } 
 
    [data-testid="stSidebar"] [data-testid="stExpander"] { 
        background: rgba(6,27,46,.35); 
        border: 1px solid rgba(255,255,255,.28); 
        border-radius: 10px; 
    } 
 
    [data-testid="stSidebar"] [data-testid="stExpander"] summary, 
    [data-testid="stSidebar"] [data-testid="stExpander"] summary * { 
        background: var(--navy) !important; 
        color: #ffffff !important; 
        font-weight: 700 !important; 
    } 
 
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover, 
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover * { 
        background: var(--blue) !important; 
        color: #ffffff !important; 
    } 
 
    [data-testid="stSidebar"] [data-testid="stFileUploader"] { 
        background: rgba(255,255,255,.1); 
        border: 1px dashed rgba(255,255,255,.7); 
        border-radius: 8px; 
        padding: 8px; 
    } 
 
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { 
        background: rgba(255,255,255,.12) !important; 
        border: 1px solid rgba(255,255,255,.5) !important; 
        border-radius: 7px !important; 
    } 
 
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button, 
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * { 
        background: #ffffff !important; 
        color: var(--navy) !important; 
        border-color: #b7d0df !important; 
        font-weight: 700 !important; 
    } 
 
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover, 
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover * { 
        background: #dff6f4 !important; 
        color: var(--navy-deep) !important; 
        border-color: var(--teal) !important; 
    } 
 
    [data-testid="stSidebar"] .stButton > button { 
        background: var(--blue) !important; 
        border-color: #8bd5d5 !important; 
        color: #ffffff !important; 
    } 
 
    [data-testid="stSidebar"] .stButton > button:hover { 
        background: var(--teal) !important; 
        border-color: #b4efea !important; 
        color: #ffffff !important; 
    } 
 
    .hero { 
        background: linear-gradient(135deg, var(--navy), var(--blue)); 
        padding: 30px; 
        border-radius: 12px; 
        margin-bottom: 25px; 
        box-shadow: 0 12px 30px rgba(18,59,93,.18); 
    } 
 
    .hero h1 { 
        color: white; 
        text-align: center; 
        font-size: 2.35rem; 
        font-weight: 800; 
        margin: 0; 
    } 
 
    .hero p { 
        color: #e6f4fa; 
        text-align: center; 
        margin-top: 10px; 
        font-size: 1rem; 
    } 
 
    .section-title { 
        background: linear-gradient(90deg, var(--navy), var(--blue)); 
        color: white; 
        padding: 11px 17px; 
        border-radius: 8px; 
        font-weight: 700; 
        margin: 20px 0 15px 0; 
    } 
 
    .empty-box { 
        background: rgba(255,255,255,.9); 
        border: 2px dashed #9bbaca; 
        border-radius: 18px; 
        padding: 45px; 
        text-align: center; 
        margin-top: 20px; 
        box-shadow: 0 8px 25px rgba(15,23,42,.06); 
    } 
 
    .empty-box h2 { 
        color: var(--text); 
        margin-bottom: 10px; 
    } 
 
    .empty-box p { 
        color: var(--muted); 
    } 
 
    .metric-card { 
        background: var(--card); 
        border: 1px solid var(--border); 
        border-radius: 10px; 
        padding: 18px; 
        min-height: 115px; 
        box-shadow: 0 5px 18px rgba(18,59,93,.08); 
    } 
 
    .metric-label { 
        color: var(--muted); 
        font-size: .72rem; 
        text-transform: uppercase; 
        letter-spacing: 1px; 
        font-weight: 700; 
    } 
 
    .metric-value { 
        color: var(--text); 
        font-size: 1.5rem; 
        font-weight: 800; 
        margin-top: 8px; 
    } 
 
    .metric-help { 
        color: #526777; 
        font-size: .75rem; 
        margin-top: 5px; 
    } 
 
    .success-box { 
        background: #e8f6ef; 
        border: 1px solid #8bcfae; 
        padding: 18px; 
        border-radius: 12px; 
        color: #065f46; 
    } 
 
    .decision-box { 
        background: linear-gradient(135deg, #e8f7fb, #f2fbfd); 
        border: 1px solid #9fd3e1; 
        border-radius: 15px; 
        padding: 25px; 
        text-align: center; 
        margin-top: 15px; 
    } 
 
    .decision-value { 
        color: var(--blue); 
        font-size: 2.4rem; 
        font-weight: 800; 
    } 
 
    /* Streamlit controls share a few base selectors, so keep their states consistent. */ 
    .stButton > button, 
    [data-testid="stDownloadButton"] > button { 
        background: var(--teal) !important; 
        color: #ffffff !important; 
        border: 1px solid #087477 !important; 
        border-radius: 8px !important; 
        font-weight: 700 !important; 
        min-height: 2.7rem; 
        box-shadow: 0 2px 5px rgba(18,59,93,.14); 
    } 
 
    .stButton > button:hover, 
    [data-testid="stDownloadButton"] > button:hover { 
        background: #087477 !important; 
        border-color: #06585b !important; 
        color: #ffffff !important; 
    } 
 
    .stButton > button:active, 
    [data-testid="stDownloadButton"] > button:active { 
        background: #06585b !important; 
        transform: translateY(1px); 
    } 
 
    .stButton > button:disabled, 
    [data-testid="stDownloadButton"] > button:disabled { 
        background: #d8e1e7 !important; 
        border-color: #b8c6d0 !important; 
        color: #526777 !important; 
        opacity: 1 !important; 
    } 
 
    [data-testid="stTextInput"] input, 
    [data-testid="stNumberInput"] input, 
    [data-testid="stMultiSelect"] [role="combobox"], 
    [data-testid="stSelectbox"] [role="combobox"] { 
        background: #ffffff !important; 
        color: var(--text) !important; 
        border-color: #9fb4c3 !important; 
    } 
 
    [data-testid="stTextInput"] input:focus, 
    [data-testid="stNumberInput"] input:focus, 
    [data-testid="stMultiSelect"] [role="combobox"]:focus, 
    [data-testid="stSelectbox"] [role="combobox"]:focus { 
        border-color: var(--teal) !important; 
        box-shadow: 0 0 0 2px rgba(21,154,156,.2) !important; 
    } 
 
    [data-testid="stTabs"] [role="tab"] { 
        color: var(--muted) !important; 
        font-weight: 700; 
    } 
 
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] { 
        color: var(--navy) !important; 
        border-bottom-color: var(--teal) !important; 
    } 
 
    [data-testid="stAlert"] { 
        color: var(--text); 
        border-radius: 8px; 
    } 
 
    [data-testid="stDataFrame"] { 
        border: 1px solid var(--border); 
        border-radius: 8px; 
    } 
 
    @media (max-width: 768px) { 
        .block-container { 
            padding: 1rem 1rem 2rem; 
        } 
 
        .hero { 
            padding: 22px 16px; 
        } 
 
        .hero h1 { 
            font-size: 1.65rem; 
        } 
    } 
    </style> 
    """, 
    unsafe_allow_html=True, 
) 
 
 
# ============================================================ 
# SQLITE DATABASE 
# ============================================================ 
def get_connection(): 
    return sqlite3.connect(DB_PATH, check_same_thread=False) 
 
 
def init_database(): 
    with get_connection() as conn: 
        conn.execute( 
            """ 
            CREATE TABLE IF NOT EXISTS inventory_data ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                Date TEXT NOT NULL, 
                Product_ID TEXT NOT NULL, 
                Demand REAL NOT NULL, 
                Inventory_Level REAL NOT NULL, 
                Lead_Time REAL NOT NULL, 
                Unit_Cost REAL NOT NULL, 
                Holding_Cost REAL NOT NULL, 
                Stockout_Cost REAL NOT NULL, 
                Ordering_Cost REAL NOT NULL 
            ) 
            """ 
        ) 
 
        conn.execute( 
            """ 
            CREATE TABLE IF NOT EXISTS training_results ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                episode INTEGER, 
                reward REAL, 
                average_reward REAL, 
                training_date TEXT 
            ) 
            """ 
        ) 
 
        conn.execute( 
            """ 
            CREATE TABLE IF NOT EXISTS evaluation_results ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                strategy TEXT, 
                total_cost REAL, 
                holding_cost REAL, 
                stockout_cost REAL, 
                ordering_cost REAL, 
                stockout_count INTEGER, 
                average_inventory REAL, 
                service_level REAL, 
                total_reward REAL, 
                evaluation_date TEXT 
            ) 
            """ 
        ) 
 
        conn.commit() 
 
 
def inventory_count(): 
    with get_connection() as conn: 
        result = conn.execute( 
            "SELECT COUNT(*) FROM inventory_data" 
        ).fetchone() 
        return int(result[0]) 
 
 
def load_inventory(): 
    with get_connection() as conn: 
        df = pd.read_sql_query( 
            """ 
            SELECT Date, Product_ID, Demand, Inventory_Level, 
                   Lead_Time, Unit_Cost, Holding_Cost, 
                   Stockout_Cost, Ordering_Cost 
            FROM inventory_data 
            ORDER BY Date, Product_ID 
            """, 
            conn, 
        ) 
 
    if not df.empty: 
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce") 
 
    return df 
 
 
def save_inventory(df): 
    clean = df.copy() 
 
    clean["Date"] = pd.to_datetime( 
        clean["Date"], errors="coerce" 
    ).dt.strftime("%Y-%m-%d") 
 
    with get_connection() as conn: 
        conn.execute("DELETE FROM inventory_data") 
        conn.execute("DELETE FROM training_results") 
        conn.execute("DELETE FROM evaluation_results") 
 
        clean.to_sql( 
            "inventory_data", 
            conn, 
            if_exists="append", 
            index=False, 
        ) 
 
        conn.commit() 
 
    if os.path.exists(MODEL_PATH): 
        os.remove(MODEL_PATH) 
 
 
def clear_all_data(): 
    with get_connection() as conn: 
        conn.execute("DELETE FROM inventory_data") 
        conn.execute("DELETE FROM training_results") 
        conn.execute("DELETE FROM evaluation_results") 
        conn.commit() 
 
    if os.path.exists(MODEL_PATH): 
        os.remove(MODEL_PATH) 
 
 
# ============================================================ 
# VALIDATION 
# ============================================================ 
def validate_dataset(df): 
    missing = [ 
        col for col in REQUIRED_COLUMNS 
        if col not in df.columns 
    ] 
 
    if missing: 
        return False, None, ( 
            "Missing required columns: " 
            + ", ".join(missing) 
        ) 
 
    work = df[REQUIRED_COLUMNS].copy() 
 
    work["Date"] = pd.to_datetime( 
        work["Date"], errors="coerce" 
    ) 
 
    numeric_columns = [ 
        "Demand", 
        "Inventory_Level", 
        "Lead_Time", 
        "Unit_Cost", 
        "Holding_Cost", 
        "Stockout_Cost", 
        "Ordering_Cost", 
    ] 
 
    for col in numeric_columns: 
        work[col] = pd.to_numeric( 
            work[col], 
            errors="coerce" 
        ) 
 
    if work["Date"].isna().all(): 
        return False, None, "Date column contains no valid dates." 
 
    if work["Product_ID"].isna().all(): 
        return False, None, "Product_ID cannot be empty." 
 
    work = work.dropna( 
        subset=["Date", "Product_ID"] 
    ) 
 
    for col in numeric_columns: 
        if work[col].isna().all(): 
            return False, None, ( 
                f"{col} does not contain valid numeric values." 
            ) 
 
        median_value = work[col].median() 
 
        if pd.isna(median_value): 
            median_value = 0 
 
        work[col] = work[col].fillna(median_value) 
 
        if (work[col] < 0).any(): 
            return False, None, ( 
                f"{col} contains negative values." 
            ) 
 
    work["Product_ID"] = work["Product_ID"].astype(str) 
 
    work = work.sort_values( 
        ["Date", "Product_ID"] 
    ).reset_index(drop=True) 
 
    return True, work, "Dataset is valid." 
 
 
# ============================================================ 
# DQN NETWORK 
# ============================================================ 
class ReplayBuffer: 
    def __init__(self, capacity=5000): 
        self.capacity = capacity 
        self.buffer = [] 
        self.position = 0 
 
    def push( 
        self, 
        state, 
        action, 
        reward, 
        next_state, 
        done, 
    ): 
        item = ( 
            np.asarray(state, dtype=np.float32), 
            int(action), 
            float(reward), 
            np.asarray(next_state, dtype=np.float32), 
            bool(done), 
        ) 
 
        if len(self.buffer) < self.capacity: 
            self.buffer.append(item) 
        else: 
            self.buffer[self.position] = item 
 
        self.position = ( 
            self.position + 1 
        ) % self.capacity 
 
    def sample(self, batch_size): 
        indices = np.random.choice( 
            len(self.buffer), 
            batch_size, 
            replace=False, 
        ) 
 
        batch = [self.buffer[i] for i in indices] 
 
        states, actions, rewards, next_states, dones = zip( 
            *batch 
        ) 
 
        return ( 
            np.asarray(states, dtype=np.float32), 
            np.asarray(actions, dtype=np.int64), 
            np.asarray(rewards, dtype=np.float32), 
            np.asarray(next_states, dtype=np.float32), 
            np.asarray(dones, dtype=np.float32), 
        ) 
 
    def __len__(self): 
        return len(self.buffer) 
 
 
class DQNNetwork(nn.Module): 
    def __init__(self): 
        super().__init__() 
 
        self.net = nn.Sequential( 
            nn.Linear(4, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, 6), 
        ) 
 
    def forward(self, x): 
        return self.net(x) 
 
 
class DQNAgent: 
    def __init__(self): 
        self.device = torch.device( 
            "cuda" 
            if torch.cuda.is_available() 
            else "cpu" 
        ) 
 
        self.policy_net = DQNNetwork().to( 
            self.device 
        ) 
 
        self.target_net = DQNNetwork().to( 
            self.device 
        ) 
 
        self.target_net.load_state_dict( 
            self.policy_net.state_dict() 
        ) 
 
        self.optimizer = optim.Adam( 
            self.policy_net.parameters(), 
            lr=0.001, 
        ) 
 
        self.memory = ReplayBuffer() 
 
        self.gamma = 0.95 
        self.epsilon = 1.0 
        self.epsilon_min = 0.05 
        self.epsilon_decay = 0.98 
 
        self.loss_fn = nn.SmoothL1Loss() 
 
    def select_action( 
        self, 
        state, 
        training=True, 
    ): 
        if ( 
            training 
            and np.random.random() < self.epsilon 
        ): 
            return np.random.randint(6) 
 
        tensor = torch.tensor( 
            state, 
            dtype=torch.float32, 
            device=self.device, 
        ).unsqueeze(0) 
 
        with torch.no_grad(): 
            q_values = self.policy_net(tensor) 
 
        return int( 
            torch.argmax(q_values, dim=1).item() 
        ) 
 
    def train_step(self): 
        if len(self.memory) < 32: 
            return 
 
        ( 
            states, 
            actions, 
            rewards, 
            next_states, 
            dones, 
        ) = self.memory.sample(32) 
 
        states = torch.tensor( 
            states, 
            dtype=torch.float32, 
            device=self.device, 
        ) 
 
        actions = torch.tensor( 
            actions, 
            dtype=torch.long, 
            device=self.device, 
        ).unsqueeze(1) 
 
        rewards = torch.tensor( 
            rewards, 
            dtype=torch.float32, 
            device=self.device, 
        ).unsqueeze(1) 
 
        next_states = torch.tensor( 
            next_states, 
            dtype=torch.float32, 
            device=self.device, 
        ) 
 
        dones = torch.tensor( 
            dones, 
            dtype=torch.float32, 
            device=self.device, 
        ).unsqueeze(1) 
 
        current_q = self.policy_net( 
            states 
        ).gather(1, actions) 
 
        with torch.no_grad(): 
            next_q = self.target_net( 
                next_states 
            ).max( 
                dim=1, 
                keepdim=True 
            )[0] 
 
            target_q = ( 
                rewards 
                + self.gamma 
                * next_q 
                * (1 - dones) 
            ) 
 
        loss = self.loss_fn( 
            current_q, 
            target_q, 
        ) 
 
        self.optimizer.zero_grad() 
        loss.backward() 
 
        torch.nn.utils.clip_grad_norm_( 
            self.policy_net.parameters(), 
            1.0, 
        ) 
 
        self.optimizer.step() 
 
    def update_target(self): 
        self.target_net.load_state_dict( 
            self.policy_net.state_dict() 
        ) 
 
    def save(self): 
        torch.save( 
            self.policy_net.state_dict(), 
            MODEL_PATH, 
        ) 
 
    def load(self): 
        state = torch.load( 
            MODEL_PATH, 
            map_location=self.device, 
        ) 
 
        self.policy_net.load_state_dict(state) 
        self.target_net.load_state_dict(state) 
 
 
# ============================================================ 
# INVENTORY ENVIRONMENT 
# ============================================================ 
class InventoryEnvironment: 
    def __init__(self, df): 
        self.df = df.reset_index( 
            drop=True 
        ) 
 
        self.max_inventory = max( 
            float(df["Inventory_Level"].max()), 
            1, 
        ) 
 
        self.max_demand = max( 
            float(df["Demand"].max()), 
            1, 
        ) 
 
        self.max_lead = max( 
            float(df["Lead_Time"].max()), 
            1, 
        ) 
 
        self.index = 0 
        self.inventory = 0 
 
    def state( 
        self, 
        inventory, 
        demand, 
        previous_demand, 
        lead_time, 
    ): 
        return np.array( 
            [ 
                inventory / self.max_inventory, 
                demand / self.max_demand, 
                previous_demand / self.max_demand, 
                lead_time / self.max_lead, 
            ], 
            dtype=np.float32, 
        ) 
 
    def reset(self): 
        self.index = 0 
 
        row = self.df.iloc[0] 
 
        self.inventory = float( 
            row["Inventory_Level"] 
        ) 
 
        return self.state( 
            self.inventory, 
            row["Demand"], 
            row["Demand"], 
            row["Lead_Time"], 
        ) 
 
    def step(self, action): 
        row = self.df.iloc[self.index] 
 
        order = float(ACTIONS[action]) 
        demand = float(row["Demand"]) 
 
        available = ( 
            self.inventory + order 
        ) 
 
        fulfilled = min( 
            available, 
            demand, 
        ) 
 
        shortage = max( 
            demand - available, 
            0, 
        ) 
 
        ending_inventory = max( 
            available - demand, 
            0, 
        ) 
 
        holding_cost = ( 
            ending_inventory 
            * float(row["Holding_Cost"]) 
        ) 
 
        stockout_cost = ( 
            shortage 
            * float(row["Stockout_Cost"]) 
        ) 
 
        ordering_cost = ( 
            float(row["Ordering_Cost"]) 
            if order > 0 
            else 0 
        ) 
 
        total_cost = ( 
            holding_cost 
            + stockout_cost 
            + ordering_cost 
        ) 
 
        reward = ( 
            fulfilled 
            * float(row["Unit_Cost"]) 
            - total_cost 
        ) 
 
        self.inventory = ending_inventory 
 
        self.index += 1 
 
        done = ( 
            self.index >= len(self.df) 
        ) 
 
        if done: 
            next_state = self.state( 
                self.inventory, 
                0, 
                demand, 
                row["Lead_Time"], 
            ) 
        else: 
            next_row = self.df.iloc[ 
                self.index 
            ] 
 
            next_state = self.state( 
                self.inventory, 
                next_row["Demand"], 
                demand, 
                next_row["Lead_Time"], 
            ) 
 
        info = { 
            "order": order, 
            "demand": demand, 
            "fulfilled": fulfilled, 
            "shortage": shortage, 
            "ending_inventory": ending_inventory, 
            "holding_cost": holding_cost, 
            "stockout_cost": stockout_cost, 
            "ordering_cost": ordering_cost, 
            "total_cost": total_cost, 
        } 
 
        return ( 
            next_state, 
            reward, 
            done, 
            info, 
        ) 
 
 
# ============================================================ 
# TRAINING 
# ============================================================ 
def train_dqn( 
    df, 
    episodes, 
    progress, 
    status, 
): 
    agent = DQNAgent() 
    history = [] 
 
    for episode in range( 
        1, 
        episodes + 1, 
    ): 
        env = InventoryEnvironment(df) 
 
        state = env.reset() 
 
        total_reward = 0 
 
        for _ in range(len(df)): 
            action = agent.select_action( 
                state, 
                training=True, 
            ) 
 
            next_state, reward, done, _ = ( 
                env.step(action) 
            ) 
 
            agent.memory.push( 
                state, 
                action, 
                reward, 
                next_state, 
                done, 
            ) 
 
            agent.train_step() 
 
            state = next_state 
            total_reward += reward 
 
            if done: 
                break 
 
        agent.epsilon = max( 
            agent.epsilon_min, 
            agent.epsilon 
            * agent.epsilon_decay, 
        ) 
 
        if episode % 10 == 0: 
            agent.update_target() 
 
        history.append( 
            float(total_reward) 
        ) 
 
        progress.progress( 
            episode / episodes 
        ) 
 
        status.write( 
            f"Training episode " 
            f"{episode}/{episodes} " 
            f"| Reward: {total_reward:,.2f}" 
        ) 
 
    agent.update_target() 
    agent.save() 
 
    with get_connection() as conn: 
        conn.execute( 
            "DELETE FROM training_results" 
        ) 
 
        running = [] 
 
        for i, reward in enumerate( 
            history, 
            start=1, 
        ): 
            running.append(reward) 
 
            avg = np.mean( 
                running[-20:] 
            ) 
 
            conn.execute( 
                """ 
                INSERT INTO training_results 
                (episode, reward, 
                 average_reward, 
                 training_date) 
                VALUES (?, ?, ?, ?) 
                """, 
                ( 
                    i, 
                    reward, 
                    avg, 
                    datetime.now().isoformat( 
                        timespec="seconds" 
                    ), 
                ), 
            ) 
 
        conn.commit() 
 
    return history 
 
 
# ============================================================ 
# STRATEGIES 
# ============================================================ 
def traditional_order( 
    inventory, 
    demand, 
    df, 
): 
    average_demand = max( 
        df["Demand"].mean(), 
        1, 
    ) 
 
    average_lead = max( 
        df["Lead_Time"].mean(), 
        1, 
    ) 
 
    reorder_point = ( 
        average_demand 
        * average_lead 
    ) 
 
    if inventory < reorder_point: 
        return max( 
            20, 
            round( 
                average_demand / 20 
            ) * 20, 
        ) 
 
    return 0 
 
 
def evaluate_strategy( 
    df, 
    strategy, 
): 
    env = InventoryEnvironment(df) 
 
    if strategy == "DQN": 
        agent = DQNAgent() 
        agent.load() 
 
    state = env.reset() 
 
    total_cost = 0 
    holding = 0 
    stockout = 0 
    ordering = 0 
    reward_total = 0 
    stockout_count = 0 
    inventory_values = [] 
    fulfilled_total = 0 
    demand_total = 0 
 
    for _ in range(len(df)): 
        row = df.iloc[env.index] 
 
        if strategy == "DQN": 
            action = agent.select_action( 
                state, 
                training=False, 
            ) 
        else: 
            order = traditional_order( 
                env.inventory, 
                row["Demand"], 
                df, 
            ) 
 
            action = int( 
                np.argmin( 
                    np.abs( 
                        ACTIONS - order 
                    ) 
                ) 
            ) 
 
        state, reward, done, info = ( 
            env.step(action) 
        ) 
 
        total_cost += info["total_cost"] 
        holding += info["holding_cost"] 
        stockout += info["stockout_cost"] 
        ordering += info["ordering_cost"] 
        reward_total += reward 
 
        inventory_values.append( 
            info["ending_inventory"] 
        ) 
 
        fulfilled_total += info["fulfilled"] 
        demand_total += info["demand"] 
 
        if info["shortage"] > 0: 
            stockout_count += 1 
 
        if done: 
            break 
 
    service_level = ( 
        fulfilled_total 
        / demand_total 
        * 100 
        if demand_total > 0 
        else 0 
    ) 
 
    return { 
        "total_cost": total_cost, 
        "holding_cost": holding, 
        "stockout_cost": stockout, 
        "ordering_cost": ordering, 
        "stockout_count": stockout_count, 
        "average_inventory": np.mean( 
            inventory_values 
        ), 
        "service_level": service_level, 
        "total_reward": reward_total, 
    } 
 
 
# ============================================================ 
# HELPERS 
# ============================================================ 
def metric( 
    label, 
    value, 
    description="", 
): 
    st.markdown( 
        f""" 
        <div class="metric-card"> 
            <div class="metric-label">{label}</div> 
            <div class="metric-value">{value}</div> 
            <div class="metric-help">{description}</div> 
        </div> 
        """, 
        unsafe_allow_html=True, 
    ) 
 
 
def section(title): 
    st.markdown( 
        f""" 
        <div class="section-title"> 
            {title} 
        </div> 
        """, 
        unsafe_allow_html=True, 
    ) 
 
 
# ============================================================ 
# INITIALIZE DATABASE 
# ============================================================ 
init_database() 
 
 
# ============================================================ 
# SIDEBAR 
# ============================================================ 
st.sidebar.markdown( 
    "# 📦 Inventory AI" 
) 
 
st.sidebar.caption( 
    "Deep Q-Network Inventory Decision Support" 
) 
 
page = st.sidebar.radio( 
    "Navigation", 
    [ 
        "Dashboard", 
        "Module 1 — Demand & Inventory Analysis", 
        "Module 2 — DQN Optimization", 
        "Module 3 — Performance Evaluation", 
    ], 
) 
 
st.sidebar.divider() 
 
count = inventory_count() 
 
if count == 0: 
    st.sidebar.warning( 
        "SQLite: No Data" 
    ) 
else: 
    st.sidebar.success( 
        "SQLite: Connected" 
    ) 
 
st.sidebar.caption( 
    f"Inventory records: {count:,}" 
) 
 
 
# ============================================================ 
# UPLOAD AREA 
# ============================================================ 
with st.sidebar.expander( 
    "📥 Dataset Management", 
    expanded=True, 
): 
 
    uploaded_file = st.file_uploader( 
        "Upload inventory CSV", 
        type=["csv"], 
        key="inventory_upload", 
    ) 
 
    if uploaded_file is not None: 
 
        if ( 
            "uploaded_file_name" 
            not in st.session_state 
            or st.session_state[ 
                "uploaded_file_name" 
            ] 
            != uploaded_file.name 
        ): 
            try: 
                uploaded_df = pd.read_csv( 
                    uploaded_file 
                ) 
 
                valid, cleaned, message = ( 
                    validate_dataset( 
                        uploaded_df 
                    ) 
                ) 
 
                st.session_state[ 
                    "uploaded_file_name" 
                ] = uploaded_file.name 
 
                st.session_state[ 
                    "uploaded_valid" 
                ] = valid 
 
                st.session_state[ 
                    "uploaded_data" 
                ] = cleaned 
 
                st.session_state[ 
                    "uploaded_message" 
                ] = message 
 
            except Exception as error: 
                st.session_state[ 
                    "uploaded_valid" 
                ] = False 
 
                st.session_state[ 
                    "uploaded_data" 
                ] = None 
 
                st.session_state[ 
                    "uploaded_message" 
                ] = str(error) 
 
        if st.session_state.get( 
            "uploaded_valid", 
            False, 
        ): 
 
            st.success( 
                "CSV validated successfully." 
            ) 
 
            st.caption( 
                f"Rows: " 
                f"{len(st.session_state['uploaded_data']):,}" 
            ) 
 
            start = st.button( 
                "🚀 START INVENTORY OPTIMIZATION", 
                type="primary", 
                use_container_width=True, 
            ) 
 
            if start: 
 
                with st.spinner( 
                    "Initializing inventory system..." 
                ): 
                    save_inventory( 
                        st.session_state[ 
                            "uploaded_data" 
                        ] 
                    ) 
 
                st.session_state[ 
                    "process_started" 
                ] = True 
 
                st.session_state[ 
                    "uploaded_file_name" 
                ] = None 
 
                st.session_state[ 
                    "uploaded_data" 
                ] = None 
 
                st.success( 
                    "Inventory system initialized successfully." 
                ) 
 
                st.rerun() 
 
        else: 
            st.error( 
                st.session_state.get( 
                    "uploaded_message", 
                    "Invalid dataset.", 
                ) 
            ) 
 
 
if st.sidebar.button( 
    "🗑️ Clear All Project Data", 
    use_container_width=True, 
): 
    clear_all_data() 
 
    for key in [ 
        "uploaded_file_name", 
        "uploaded_data", 
        "uploaded_valid", 
        "uploaded_message", 
        "process_started", 
    ]: 
        st.session_state.pop( 
            key, 
            None, 
        ) 
 
    st.rerun() 
 
 
# ============================================================ 
# LOAD DATABASE 
# ============================================================ 
df = load_inventory() 
 
 
# ============================================================ 
# NO DATA SCREEN 
# ============================================================ 
if df.empty: 
 
    st.markdown( 
        """ 
        <div class="hero"> 
            <h1> 
                REINFORCEMENT LEARNING-BASED<br> 
                INVENTORY OPTIMIZATION 
            </h1> 
            <p> 
                DQN-Powered Demand Analysis, 
                Optimization and Performance Evaluation 
            </p> 
        </div> 
        """, 
        unsafe_allow_html=True, 
    ) 
 
    st.markdown( 
        """ 
        <div class="empty-box"> 
            <h2>📦 No Inventory Data Available</h2> 
            <p> 
                The inventory database is currently empty. 
                Upload your CSV dataset from the sidebar. 
            </p> 
            <p> 
                After validation, click 
                <b>START INVENTORY OPTIMIZATION</b> 
                to initialize the system. 
            </p> 
        </div> 
        """, 
        unsafe_allow_html=True, 
    ) 
 
    section( 
        "📋 Required Dataset Structure" 
    ) 
 
    st.dataframe( 
        pd.DataFrame( 
            columns=REQUIRED_COLUMNS 
        ), 
        use_container_width=True, 
        hide_index=True, 
    ) 
 
    st.info( 
        "No sample or fake inventory data is " 
        "automatically inserted into SQLite." 
    ) 
 
    st.stop() 
 
 
# ============================================================ 
# COMMON METRICS 
# ============================================================ 
total_demand = df["Demand"].sum() 
 
average_inventory = ( 
    df["Inventory_Level"].mean() 
) 
 
current_inventory = ( 
    df.sort_values("Date") 
    ["Inventory_Level"] 
    .iloc[-1] 
) 
 
stockouts = int( 
    ( 
        df["Demand"] 
        > df["Inventory_Level"] 
    ).sum() 
) 
 
service_level = ( 
    np.minimum( 
        df["Demand"], 
        df["Inventory_Level"], 
    ).sum() 
    / total_demand 
    * 100 
    if total_demand > 0 
    else 0 
) 
 
estimated_cost = ( 
    df["Inventory_Level"] 
    * df["Holding_Cost"] 
).sum() + ( 
    np.maximum( 
        df["Demand"] 
        - df["Inventory_Level"], 
        0, 
    ) 
    * df["Stockout_Cost"] 
).sum() 
 
 
# ============================================================ 
# HEADER 
# ============================================================ 
st.markdown( 
    """ 
    <div class="hero"> 
        <h1> 
            REINFORCEMENT LEARNING-BASED<br> 
            INVENTORY OPTIMIZATION 
        </h1> 
        <p> 
            DQN-Powered Demand Analysis, 
            Optimization and Performance Evaluation 
        </p> 
    </div> 
    """, 
    unsafe_allow_html=True, 
) 
 
 
# ============================================================ 
# DASHBOARD 
# ============================================================ 
if page == "Dashboard": 
 
    section( 
        "📊 Inventory Performance Overview" 
    ) 
 
    cols = st.columns(5) 
 
    with cols[0]: 
        metric( 
            "Total Demand", 
            f"{total_demand:,.0f}", 
            "Units", 
        ) 
 
    with cols[1]: 
        metric( 
            "Current Inventory", 
            f"{current_inventory:,.0f}", 
            "Latest record", 
        ) 
 
    with cols[2]: 
        metric( 
            "Average Inventory", 
            f"{average_inventory:,.1f}", 
            "Across records", 
        ) 
 
    with cols[3]: 
        metric( 
            "Service Level", 
            f"{service_level:.1f}%", 
            "Demand fulfillment", 
        ) 
 
    with cols[4]: 
        metric( 
            "Stock-outs", 
            f"{stockouts:,}", 
            "Demand > inventory", 
        ) 
 
    c1, c2 = st.columns(2) 
 
    with c1: 
        section("📈 Demand Trend") 
 
        trend = ( 
            df.groupby("Date")["Demand"] 
            .sum() 
            .reset_index() 
        ) 
 
        fig = px.line( 
            trend, 
            x="Date", 
            y="Demand", 
            markers=True, 
        ) 
 
        fig.update_layout( 
            height=360 
        ) 
 
        st.plotly_chart( 
            fig, 
            use_container_width=True, 
        ) 
 
    with c2: 
        section("📦 Inventory Trend") 
 
        trend = ( 
            df.groupby("Date") 
            ["Inventory_Level"] 
            .mean() 
            .reset_index() 
        ) 
 
        fig = px.line( 
            trend, 
            x="Date", 
            y="Inventory_Level", 
            markers=True, 
        ) 
 
        fig.update_layout( 
            height=360 
        ) 
 
        st.plotly_chart( 
            fig, 
            use_container_width=True, 
        ) 
 
    section( 
        "🤖 Artificial Intelligence Status" 
    ) 
 
    cols = st.columns(3) 
 
    with cols[0]: 
        metric( 
            "Algorithm", 
            "DQN", 
            "Deep Q-Network", 
        ) 
 
    with cols[1]: 
        metric( 
            "Model Status", 
            "Trained" 
            if os.path.exists(MODEL_PATH) 
            else "Not Trained", 
            "PyTorch model", 
        ) 
 
    with cols[2]: 
        metric( 
            "SQLite Records", 
            f"{len(df):,}", 
            "Persistent inventory data", 
        ) 
 
    section("💰 Current Inventory Cost") 
 
    metric( 
        "Estimated Cost", 
        f"${estimated_cost:,.2f}", 
        "Holding + stockout cost", 
    ) 
 
 
# ============================================================ 
# MODULE 1 
# ============================================================ 
elif page == "Module 1 — Demand & Inventory Analysis": 
 
    section( 
        "📊 Module 1 — Demand & Inventory Analysis" 
    ) 
 
    products = sorted( 
        df["Product_ID"] 
        .astype(str) 
        .unique() 
    ) 
 
    selected = st.multiselect( 
        "Select Products", 
        products, 
        default=products, 
    ) 
 
    filtered = df[ 
        df["Product_ID"] 
        .astype(str) 
        .isin(selected) 
    ].copy() 
 
    if filtered.empty: 
        st.warning( 
            "No records selected." 
        ) 
        st.stop() 
 
    cols = st.columns(5) 
 
    with cols[0]: 
        metric( 
            "Demand", 
            f"{filtered['Demand'].sum():,.0f}", 
            "Total", 
        ) 
 
    with cols[1]: 
        metric( 
            "Average Demand", 
            f"{filtered['Demand'].mean():,.1f}", 
            "Per record", 
        ) 
 
    with cols[2]: 
        metric( 
            "Average Inventory", 
            f"{filtered['Inventory_Level'].mean():,.1f}", 
            "Per record", 
        ) 
 
    with cols[3]: 
        metric( 
            "Lead Time", 
            f"{filtered['Lead_Time'].mean():,.1f}", 
            "Days", 
        ) 
 
    with cols[4]: 
        metric( 
            "Stock-outs", 
            f"{( 
                filtered['Demand'] 
                > filtered['Inventory_Level'] 
            ).sum():,}", 
            "Records", 
        ) 
 
    c1, c2 = st.columns(2) 
 
    with c1: 
        section("📈 Demand Analysis") 
 
        trend = ( 
            filtered.groupby("Date") 
            ["Demand"] 
            .sum() 
            .reset_index() 
        ) 
 
        fig = px.line( 
            trend, 
            x="Date", 
            y="Demand", 
            markers=True, 
        ) 
 
        st.plotly_chart( 
            fig, 
            use_container_width=True, 
        ) 
 
    with c2: 
        section("📦 Inventory Analysis") 
 
        trend = ( 
            filtered.groupby("Date") 
            ["Inventory_Level"] 
            .mean() 
            .reset_index() 
        ) 
 
        fig = px.line( 
            trend, 
            x="Date", 
            y="Inventory_Level", 
            markers=True, 
        ) 
 
        st.plotly_chart( 
            fig, 
            use_container_width=True, 
        ) 
 
    section("🏷️ Product Demand") 
 
    product_demand = ( 
        filtered.groupby( 
            "Product_ID" 
        )["Demand"] 
        .sum() 
        .reset_index() 
        .sort_values( 
            "Demand", 
            ascending=False, 
        ) 
    ) 
 
    fig = px.bar( 
        product_demand, 
        x="Product_ID", 
        y="Demand", 
    ) 
 
    st.plotly_chart( 
        fig, 
        use_container_width=True, 
    ) 
 
    section("📋 Inventory Data") 
 
    st.dataframe( 
        filtered, 
        use_container_width=True, 
        hide_index=True, 
    ) 
 
 
# ============================================================ 
# MODULE 2 
# ============================================================ 
elif page == "Module 2 — DQN Optimization": 
 
    section( 
        "🤖 Module 2 — DQN Optimization" 
    ) 
 
    st.info( 
        "The DQN learns an inventory ordering policy " 
        "using inventory level, demand, previous demand " 
        "and lead time as the state." 
    ) 
 
    cols = st.columns(4) 
 
    with cols[0]: 
        metric( 
            "Algorithm", 
            "DQN", 
            "Deep Q-Network", 
        ) 
 
    with cols[1]: 
        metric( 
            "State Size", 
            "4", 
            "Inventory / demand / previous demand / lead time", 
        ) 
 
    with cols[2]: 
        metric( 
            "Actions", 
            "6", 
            "0–100 units", 
        ) 
 
    with cols[3]: 
        metric( 
            "Device", 
            "GPU" 
            if torch.cuda.is_available() 
            else "CPU", 
            "PyTorch", 
        ) 
 
    section("⚙️ Train DQN") 
 
    episodes = st.number_input( 
        "Training Episodes", 
        min_value=10, 
        max_value=500, 
        value=50, 
        step=10, 
    ) 
 
    train_button = st.button( 
        "🚀 TRAIN DQN MODEL", 
        type="primary", 
        use_container_width=True, 
    ) 
 
    if train_button: 
 
        progress = st.progress(0) 
        status = st.empty() 
 
        try: 
 
            history = train_dqn( 
                df, 
                int(episodes), 
                progress, 
                status, 
            ) 
 
            st.success( 
                "DQN training completed successfully." 
            ) 
 
        except Exception as error: 
 
            st.error( 
                f"Training failed: {error}" 
            ) 
 
    if os.path.exists( 
        MODEL_PATH 
    ): 
 
        section( 
            "🎯 AI Inventory Decision" 
        ) 
 
        c1, c2, c3, c4 = st.columns(4) 
 
        with c1: 
            current = st.number_input( 
                "Current Inventory", 
                min_value=0.0, 
                value=float( 
                    current_inventory 
                ), 
            ) 
 
        with c2: 
            demand = st.number_input( 
                "Expected Demand", 
                min_value=0.0, 
                value=float( 
                    df["Demand"].mean() 
                ), 
            ) 
 
        with c3: 
            previous = st.number_input( 
                "Previous Demand", 
                min_value=0.0, 
                value=float( 
                    df["Demand"].iloc[-1] 
                ), 
            ) 
 
        with c4: 
            lead = st.number_input( 
                "Lead Time", 
                min_value=0.0, 
                value=float( 
                    df["Lead_Time"].mean() 
                ), 
            ) 
 
        if st.button( 
            "✨ GENERATE OPTIMAL ORDER", 
            type="primary", 
            use_container_width=True, 
        ): 
 
            agent = DQNAgent() 
            agent.load() 
 
            env = InventoryEnvironment( 
                df 
            ) 
 
            state = env.state( 
                current, 
                demand, 
                previous, 
                lead, 
            ) 
 
            tensor = torch.tensor( 
                state, 
                dtype=torch.float32, 
                device=agent.device, 
            ).unsqueeze(0) 
 
            with torch.no_grad(): 
                q_values = ( 
                    agent.policy_net( 
                        tensor 
                    ) 
                    .cpu() 
                    .numpy()[0] 
                ) 
 
            best = int( 
                np.argmax(q_values) 
            ) 
 
            recommendation = ACTIONS[ 
                best 
            ] 
 
            st.markdown( 
                f""" 
                <div class="decision-box"> 
                    <div class="metric-label"> 
                        DQN Recommended Order 
                    </div> 
                    <div class="decision-value"> 
                        {recommendation:,.0f} units 
                    </div> 
                </div> 
                """, 
                unsafe_allow_html=True, 
            ) 
 
            q_table = pd.DataFrame( 
                { 
                    "Order Quantity": ACTIONS.astype( 
                        int 
                    ), 
                    "Q-value": q_values, 
                } 
            ) 
 
            st.dataframe( 
                q_table, 
                use_container_width=True, 
                hide_index=True, 
            ) 
 
 
# ============================================================ 
# MODULE 3 
# ============================================================ 
elif page == "Module 3 — Performance Evaluation": 
 
    section( 
        "📈 Module 3 — Performance Evaluation" 
    ) 
 
    if not os.path.exists( 
        MODEL_PATH 
    ): 
 
        st.warning( 
            "Please train the DQN model in " 
            "Module 2 first." 
        ) 
 
        st.stop() 
 
    st.info( 
        "This module compares the traditional " 
        "reorder-point strategy against the " 
        "trained DQN strategy." 
    ) 
 
    if st.button( 
        "▶ RUN TRADITIONAL VS DQN", 
        type="primary", 
        use_container_width=True, 
    ): 
 
        with st.spinner( 
            "Evaluating inventory strategies..." 
        ): 
 
            traditional = ( 
                evaluate_strategy( 
                    df, 
                    "Traditional", 
                ) 
            ) 
 
            dqn = evaluate_strategy( 
                df, 
                "DQN", 
            ) 
 
            results = pd.DataFrame( 
                [ 
                    { 
                        "Strategy": "Traditional", 
                        **traditional, 
                    }, 
                    { 
                        "Strategy": "DQN", 
                        **dqn, 
                    }, 
                ] 
            ) 
 
            with get_connection() as conn: 
 
                conn.execute( 
                    "DELETE FROM evaluation_results" 
                ) 
 
                for _, row in results.iterrows(): 
 
                    conn.execute( 
                        """ 
                        INSERT INTO evaluation_results 
                        ( 
                            strategy, 
                            total_cost, 
                            holding_cost, 
                            stockout_cost, 
                            ordering_cost, 
                            stockout_count, 
                            average_inventory, 
                            service_level, 
                            total_reward, 
                            evaluation_date 
                        ) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) 
                        """, 
                        ( 
                            row["Strategy"], 
                            row["total_cost"], 
                            row["holding_cost"], 
                            row["stockout_cost"], 
                            row["ordering_cost"], 
                            row["stockout_count"], 
                            row["average_inventory"], 
                            row["service_level"], 
                            row["total_reward"], 
                            datetime.now().isoformat( 
                                timespec="seconds" 
                            ), 
                        ), 
                    ) 
 
                conn.commit() 
 
        st.success( 
            "Evaluation completed successfully." 
        ) 
 
    with get_connection() as conn: 
 
        evaluation = pd.read_sql_query( 
            """ 
            SELECT strategy, 
                   total_cost, 
                   holding_cost, 
                   stockout_cost, 
                   ordering_cost, 
                   stockout_count, 
                   average_inventory, 
                   service_level, 
                   total_reward 
            FROM evaluation_results 
            ORDER BY id 
            """, 
            conn, 
        ) 
 
    if evaluation.empty: 
 
        st.info( 
            "Click the evaluation button to " 
            "generate the comparison." 
        ) 
 
        st.stop() 
 
    section( 
        "⚖️ Traditional vs DQN" 
    ) 
 
    display = evaluation.copy() 
 
    display.columns = [ 
        "Strategy", 
        "Total Cost", 
        "Holding Cost", 
        "Stockout Cost", 
        "Ordering Cost", 
        "Stock-outs", 
        "Average Inventory", 
        "Service Level (%)", 
        "Total Reward", 
    ] 
 
    st.dataframe( 
        display, 
        use_container_width=True, 
        hide_index=True, 
    ) 
 
    c1, c2 = st.columns(2) 
 
    with c1: 
 
        section( 
            "💰 Cost Comparison" 
        ) 
 
        cost_data = evaluation[ 
            [ 
                "strategy", 
                "total_cost", 
                "holding_cost", 
                "stockout_cost", 
                "ordering_cost", 
            ] 
        ].melt( 
            id_vars="strategy", 
            var_name="Cost Type", 
            value_name="Cost", 
        ) 
 
        fig = px.bar( 
            cost_data, 
            x="Cost Type", 
            y="Cost", 
            color="strategy", 
            barmode="group", 
        ) 
 
        st.plotly_chart( 
            fig, 
            use_container_width=True, 
        ) 
 
    with c2: 
 
        section( 
            "📊 Operational Comparison" 
        ) 
 
        operational = evaluation[ 
            [ 
                "strategy", 
                "stockout_count", 
                "average_inventory", 
                "service_level", 
            ] 
        ].melt( 
            id_vars="strategy", 
            var_name="Metric", 
            value_name="Value", 
        ) 
 
        fig = px.bar( 
            operational, 
            x="Metric", 
            y="Value", 
            color="strategy", 
            barmode="group", 
        ) 
 
        st.plotly_chart( 
            fig, 
            use_container_width=True, 
        ) 
 
    traditional_cost = float( 
        evaluation[ 
            evaluation["strategy"] 
            == "Traditional" 
        ]["total_cost"].iloc[0] 
    ) 
 
    dqn_cost = float( 
        evaluation[ 
            evaluation["strategy"] 
            == "DQN" 
        ]["total_cost"].iloc[0] 
    ) 
 
    if traditional_cost > 0: 
 
        improvement = ( 
            ( 
                traditional_cost 
                - dqn_cost 
            ) 
            / traditional_cost 
            * 100 
        ) 
 
        if dqn_cost < traditional_cost: 
 
            st.success( 
                f"DQN achieved " 
                f"{improvement:.2f}% lower " 
                f"total cost than the " 
                f"traditional strategy." 
            ) 
 
        else: 
 
            st.warning( 
                f"DQN total cost was " 
                f"{abs(improvement):.2f}% higher " 
                f"than the traditional strategy " 
                f"on this evaluation dataset." 
            ) 