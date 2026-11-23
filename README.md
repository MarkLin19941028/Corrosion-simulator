# Corrosion Simulator（半導體濕式清洗腐蝕模擬器）

本應用程式是一個結合物理電化學模型與機器學習（Physics-Informed & ML-Assisted）的腐蝕模擬平台，用於評估半導體製程中互連層、阻障層、襯墊層與閘極材料（核心金屬 `Cu, Co, W, Ru, Ti, Ta, Al, Mo` 及擴充材料如 `TiN, Fe, Cr, Ni, Pt, Pd, Au` 等）在濕式清洗液中的單一金屬腐蝕、雙金屬伽凡尼耦合（Galvanic Coupling）加速效應、隨時間演變的厚度損失風險，以及多組清洗配方的綜合評比。

## 目錄

- [一、系統運作原理簡述](#一系統運作原理簡述)
- [二、系統需求與相依套件](#二系統需求與相依套件)
- [三、完整指令速查表（macOS 與 Windows）](#三完整指令速查表macos-與-windows)
  - [1. 環境建立與套件安裝](#1-環境建立與套件安裝)
  - [2. 啟動與操作網頁應用程式（Streamlit App）](#2-啟動與操作網頁應用程式streamlit-app)
  - [3. 資料整合與離線模型預建](#3-資料整合與離線模型預建)
  - [4. 自動化測試與驗證](#4-自動化測試與驗證)
  - [5. 示範與分析用 Jupyter Notebook](#5-示範與分析用-jupyter-notebook)
- [四、新手快速上手操作指南](#四新手快速上手操作指南)
  - [步驟 1：啟動應用程式](#步驟-1啟動應用程式)
  - [步驟 2：在左側邊欄設定模擬條件](#步驟-2在左側邊欄設定模擬條件)
  - [步驟 3：檢視右側四個分析分頁（指標、表格、圖表與名詞說明）](#步驟-3檢視右側四個分析分頁指標表格圖表與名詞說明)
  - [步驟 4：匯入自己的實驗資料（選用）](#步驟-4匯入自己的實驗資料選用)
- [五、常見問題與排錯指南](#五常見問題與排錯指南)
- [六、示範 Notebook 與進階文件](#六示範-notebook-與進階文件)

---

## 一、系統運作原理簡述

本專案採用「**離線資料整合與模型預建**」搭配「**即時互動模擬介面**」的兩段式架構：

1. **原始資料整合與離線預建（`data/raw/` → `data/processed/`）**
   - `corrosion_simulator/database.py` 會讀取 `data/raw/` 下的電化學、材料物理性質與溶液性質 CSV 檔，自動將不同參考電極（SHE、Ag/AgCl、SCE、MSE）統一換算為 `V vs. SHE`，並標準化腐蝕電流密度（`µA/cm²`）與 Tafel 斜率（正值 `mV/dec`）。
   - `corrosion_simulator/build_artifacts.py` 會將整理後的資料匯出為高速讀取的 Parquet 表與預訓練機器學習模型（`.joblib`），並寫入 `manifest.json` 記錄原始檔案指紋。若 `data/raw/` 為空，系統會自動回退至內建種子資料（`seed_data.py`），確保開箱即用。
2. **電化學參數來源（三擇一）**
   - **ML Prediction（預設）**：透過 `HybridMLPredictor` 預測 (<i>E</i><sub>corr</sub>, log<sub>10</sub> <i>i</i><sub>corr</sub>, <i>β</i><sub>a</sub>, <i>β</i><sub>c</sub>)，並以對數空間 Monte Carlo 傳播不確定性，同時判定適用域信心等級（`High / Medium / Low`）。
   - **Database Lookup**：在整理後的資料庫中進行最近鄰暴露條件查詢。
   - **Manual Override**：由使用者直接輸入自訂電化學參數。
3. **物理與伽凡尼耦合引擎**
   - 單一金屬透過 Wagner–Traud 四參數極化模型與 Faraday 定律計算初始腐蝕速率 CR<sub>0</sub>（`nm/min` 與 `mm/yr`）。
   - 雙金屬耦合使用 Brent 數值求根法（`scipy.optimize.brentq`）解面積加權電流守恆方程式，求得混合電位 <i>E</i><sub>galv</sub>、伽凡尼電流密度 <i>i</i><sub>galv</sub> 與加速倍率。
4. **時間動力學、風險評估與報告匯出**
   - 以 Linear、Passivation（自限鈍化）或 Power-Law（擴散/緩蝕）模型模擬腐蝕深度 <i>d</i>(<i>t</i>)，計算達到臨界厚度門檻 <i>d</i><sub>crit</sub> 所需時間（Time to Threshold）與厚度損失風險分級，並支援 2–5 組清洗配方並排比較及六種格式報告下載。

---

## 二、系統需求與相依套件

- **Python 版本**：`Python 3.11` 以上（建議 `3.11`–`3.13`）
- **核心必裝函式庫**（定義於 [requirements.txt](requirements.txt)）：

| 函式庫 | 建議最低版本 | 用途說明 |
| :--- | :--- | :--- |
| `numpy` | `>=2.0.0` | 數值運算與陣列計算核心 |
| `pandas` | `>=2.2.2` | 電化學資料表處理、清理與 Parquet 讀寫（相容 NumPy 2.x） |
| `pyarrow` | `>=16.0.0` | `pandas` 讀寫 `.parquet` 預建資料表所需的底層引擎 |
| `scipy` | `>=1.13.0` | 伽凡尼混合電位求根計算（`scipy.optimize.brentq`） |
| `scikit-learn` | `>=1.5.0` | 機器學習模型訓練、交叉驗證與評估指標（含 `root_mean_squared_error`） |
| `joblib` | `>=1.4.0` | 預訓練 ML 模型（`.joblib`）序列化儲存與快速載入 |
| `matplotlib` | `>=3.9.0` | 產生高解析度靜態圖表與 PNG / SVG / PDF 技術報告 |
| `plotly` | `>=5.22.0` | 網頁互動式極化曲線、動力學曲線、配方長條圖與雷達圖 |
| `streamlit` | `>=1.50.0` | 互動式網頁應用程式框架（本專案使用 `width="stretch"` 版面 API） |

> 若需執行 `notebooks/` 內的教學與分析 Notebook，請額外安裝 `jupyterlab>=4.2.0` 與 `nbconvert>=7.16.0`。

---

## 三、完整指令速查表（macOS 與 Windows）

> 以下所有指令皆預設在**專案根目錄**（包含 `app.py` 的資料夾）執行。Windows 指令適用於 **PowerShell**（建議）；若使用傳統 `cmd.exe`，可直接沿用 `.venv\Scripts\...` 路徑。

### 1. 環境建立與套件安裝

| 用途 | macOS（Terminal / zsh） | Windows（PowerShell） |
| :--- | :--- | :--- |
| 檢查 Python 版本（需 `>=3.11`） | `python3 --version` | `py -3 --version` 或 `python --version` |
| 建立虛擬環境 `.venv` | `python3 -m venv .venv` | `py -3 -m venv .venv` |
| 啟用虛擬環境（選用） | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| 升級 `pip`（建議） | `.venv/bin/python -m pip install --upgrade pip` | `.venv\Scripts\python.exe -m pip install --upgrade pip` |
| 安裝所有必裝套件 | `.venv/bin/pip install -r requirements.txt` | `.venv\Scripts\pip.exe install -r requirements.txt` |
| 以可編輯模式安裝本專案（選用） | `.venv/bin/pip install -e .` | `.venv\Scripts\pip.exe install -e .` |
| 安裝 Jupyter（選用，跑 Notebook 用） | `.venv/bin/pip install jupyterlab nbconvert` | `.venv\Scripts\pip.exe install jupyterlab nbconvert` |
| 離開虛擬環境（若先前有啟用） | `deactivate` | `deactivate` |

### 2. 啟動與操作網頁應用程式（Streamlit App）

| 用途 | macOS（Terminal / zsh） | Windows（PowerShell） |
| :--- | :--- | :--- |
| 啟動互動式應用程式（預設埠 `8501`） | `.venv/bin/streamlit run app.py` | `.venv\Scripts\streamlit.exe run app.py` |
| 使用 `python -m` 啟動（相容性最高） | `.venv/bin/python -m streamlit run app.py` | `.venv\Scripts\python.exe -m streamlit run app.py` |
| 指定其他埠號啟動（例如 `8502`） | `.venv/bin/streamlit run app.py --server.port 8502` | `.venv\Scripts\streamlit.exe run app.py --server.port 8502` |
| 啟動但不自動彈出瀏覽器（Headless） | `.venv/bin/streamlit run app.py --server.headless true` | `.venv\Scripts\streamlit.exe run app.py --server.headless true` |
| 停止運行中的應用程式 | 在終端機按 `Ctrl + C` | 在終端機按 `Ctrl + C` |

### 3. 資料整合與離線模型預建

> 每當您新增或修改 `data/raw/` 內的 CSV 檔案時，請執行此類指令以更新 `data/processed/` 中的 Parquet 資料表與 `.joblib` 模型檔。

| 用途 | macOS（Terminal / zsh） | Windows（PowerShell） |
| :--- | :--- | :--- |
| 標準重建（更新資料表 + 預建預設 3 種模型） | `.venv/bin/python -m corrosion_simulator.build_artifacts` | `.venv\Scripts\python.exe -m corrosion_simulator.build_artifacts` |
| 預建所有 6 種 ML 模型家族 | `.venv/bin/python -m corrosion_simulator.build_artifacts --models all` | `.venv\Scripts\python.exe -m corrosion_simulator.build_artifacts --models all` |
| 指定預建特定模型（逗號分隔） | `.venv/bin/python -m corrosion_simulator.build_artifacts --models random_forest,extra_trees` | `.venv\Scripts\python.exe -m corrosion_simulator.build_artifacts --models random_forest,extra_trees` |
| 僅輸出 Parquet（略過 CSV 鏡像以加快速度） | `.venv/bin/python -m corrosion_simulator.build_artifacts --no-csv` | `.venv\Scripts\python.exe -m corrosion_simulator.build_artifacts --no-csv` |
| 嚴格驗證模式（有異常資料或缺少物理性質即報錯） | `.venv/bin/python -m corrosion_simulator.build_artifacts --strict` | `.venv\Scripts\python.exe -m corrosion_simulator.build_artifacts --strict` |
| 指定自訂原始／輸出資料夾 | `.venv/bin/python -m corrosion_simulator.build_artifacts --raw-dir data/raw --out-dir data/processed` | `.venv\Scripts\python.exe -m corrosion_simulator.build_artifacts --raw-dir data/raw --out-dir data/processed` |
| 調整高斯過程（GP）最大訓練樣本數 | `.venv/bin/python -m corrosion_simulator.build_artifacts --models all --gp-max-samples 2500` | `.venv\Scripts\python.exe -m corrosion_simulator.build_artifacts --models all --gp-max-samples 2500` |
| 查看所有可用參數說明 | `.venv/bin/python -m corrosion_simulator.build_artifacts --help` | `.venv\Scripts\python.exe -m corrosion_simulator.build_artifacts --help` |

> 支援的 `--models` 名稱：`random_forest`、`extra_trees`、`hist_gradient_boosting`（以上三者為預設）、`gaussian_process`、`rf_gp_ensemble`、`gradient_boosting`，或輸入 `all`。

### 4. 自動化測試與驗證

| 用途 | macOS（Terminal / zsh） | Windows（PowerShell） |
| :--- | :--- | :--- |
| 執行完整單元與整合測試 | `.venv/bin/python -m unittest discover -s tests -v` | `.venv\Scripts\python.exe -m unittest discover -s tests -v` |
| 僅執行資料庫整合測試 | `.venv/bin/python -m unittest tests/test_database_harmonization.py -v` | `.venv\Scripts\python.exe -m unittest tests/test_database_harmonization.py -v` |
| 僅執行 Streamlit 平台介面測試 | `.venv/bin/python -m unittest tests/test_streamlit_platform.py -v` | `.venv\Scripts\python.exe -m unittest tests/test_streamlit_platform.py -v` |

### 5. 示範與分析用 Jupyter Notebook

| 用途 | macOS（Terminal / zsh） | Windows（PowerShell） |
| :--- | :--- | :--- |
| 開啟互動式 JupyterLab | `.venv/bin/jupyter lab notebooks/` | `.venv\Scripts\jupyter.exe lab notebooks/` |
| 命令列無頭模式重新執行單一 Notebook | `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration_and_cleaning.ipynb` | `.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks/01_data_exploration_and_cleaning.ipynb` |

---

## 四、新手快速上手操作指南

### 步驟 1：啟動應用程式
完成套件安裝後，執行啟動指令，瀏覽器會自動開啟 `http://localhost:8501`。

### 步驟 2：在左側邊欄設定模擬條件
1. **Metal / Material Selection**：選擇主材料（Primary Material），標籤會顯示該材料在資料庫中的樣本數 `n=...`；若要分析雙金屬腐蝕，請選擇副材料（Secondary Material）與陰陽極面積比（<i>A</i><sub>noble</sub> / <i>A</i><sub>active</sub>）。
2. **Solution & Operating Conditions**：選擇氧化劑（如 `H2O2`）、酸/錯合劑、緩蝕劑（如 `BTA`）的種類與濃度，並調整 `pH` 與操作溫度（`°C`）。
3. **Immersion Time & Kinetics**：設定浸泡時間（秒）、臨界厚度門檻 <i>d</i><sub>crit</sub>（`nm`），以及動力學模型（預設 `Auto` 會依 Pourbaix 穩定區自動挑選，也可手動切換 Linear / Passivation / Power-Law）。
4. **Parameter Source**：預設為 `ML Prediction`（可自由切換 6 種演算法）；若要直接查表可選 `Database Lookup`，若已有實驗數據則可選 `Manual Override` 手動輸入。

### 步驟 3：檢視右側四個分析分頁（指標、表格、圖表與名詞說明）

#### 分頁 1：`Single Metal Analysis`（單一金屬分析）
評估單一材料在指定清洗液下的電化學行為、腐蝕速率與時間風險。頁面頂部會顯示 **Confidence Tier** 適用域信心橫幅，可展開查看警告與實驗建議。

- **頂部關鍵指標卡（4 項 KPI）**：
  - `Initial Corrosion Rate (CR₀)`：依 Faraday 定律換算的初始腐蝕速率（`nm/min` 與 `mm/yr`）。
  - `Corrosion Depth @ t`：在設定浸泡時間結束時的累積腐蝕深度（`nm`），下方對照門檻 `Threshold: d_crit`。
  - `Time to Threshold`：腐蝕深度達到臨界厚度 `d_crit` 所需的時間（含 90% Monte Carlo 信賴區間）；若在模擬範圍內不會達到則顯示 `∞ (Not reached)`。
  - `Thickness Loss Risk`：綜合腐蝕深度與時間的風險分級（`Low / Moderate / High / Critical`）與連續風險分數 `R(t)`。
- **表格：`Electrochemical Parameters & Uncertainty (±1σ)`**
  - 列出 4 個核心電化學參數（`E_corr`、`i_corr`、`beta_a`、`beta_c`）的預測／查詢／手動輸入數值、單位、±1σ 不確定性與參數來源。
- **互動圖表**：
  1. **極化曲線（`Polarization Curve (Tafel Plot)`）**：橫軸為電位 <i>E</i>（V vs. SHE），縱軸為電流密度對數 log<sub>10</sub>(|<i>i</i>|)。可同時觀察陽極氧化分支、陰極還原分支與總淨電流曲線，曲線尖點即為腐蝕電位 <i>E</i><sub>corr</sub> 與腐蝕電流密度 <i>i</i><sub>corr</sub>。
  2. **時間動力學與風險曲線（`Time-Resolved Kinetics & Thickness Loss Risk`）**：呈現腐蝕深度 <i>d</i>(<i>t</i>) 隨時間的變化軌跡、90% 不確定性陰影區間，以及水平虛線標示的臨界厚度門檻 <i>d</i><sub>crit</sub>。可藉此判斷該配方在製程時間內是否會超過安全上限。

#### 分頁 2：`Galvanic Analysis`（雙金屬伽凡尼耦合分析）
當左側邊欄有選取「副材料（Secondary Material）」時啟用，用來評估兩種不同材料接觸同一清洗液時，電位較低的主動金屬（陽極）被加速腐蝕的程度。

- **頂部關鍵指標卡（4 項 KPI）**：
  - `Active / Noble Metal`：自動判定哪一個是**主動金屬（Active，陽極、被腐蝕端）**、哪一個是**貴金屬（Noble，陰極、受保護端）**，並顯示電位差驅動力風險（`|ΔE|`）。
  - `Mixed Potential (E_galv)`：耦合後的混合電位，以及流經耦合界面的伽凡尼電流密度 `i_galv`（`µA/cm²`）。
  - `Acceleration Factor`：耦合後的腐蝕加速倍率（`i_galv / i_corr,active`）與加速後的初始腐蝕速率 `CR₀`。
  - `Coupled Time to Threshold`：耦合加速後達到 `d_crit` 的時間，下方對照未耦合（Uncoupled）時的達標時間，可一眼看出壽命縮短幅度。
- **表格 1：`Uncoupled vs. Galvanically Coupled Summary`**
  - 將主動金屬「未耦合」與「耦合後」的 CR<sub>0</sub>、累積腐蝕深度、達門檻時間、風險分數與風險等級並排對照。
- **互動圖表**：
  1. **雙金屬疊加極化圖（`Galvanic Polarization Curve`）**：將主動金屬的陽極曲線與經面積比（<i>A</i><sub>noble</sub> / <i>A</i><sub>active</sub>）加權後的貴金屬陰極曲線畫在同一張圖上，兩曲線交點即為混合電位 <i>E</i><sub>galv</sub> 與伽凡尼電流密度 <i>i</i><sub>galv</sub>。陰極面積越大，交點越往高電流方向移動、加速越顯著。
  2. **耦合前後動力學比較圖（`Uncoupled vs. Galvanically Coupled Kinetics`）**：比較同一材料在「單獨存在」與「與貴金屬耦合」兩種情況下的腐蝕深度成長曲線，直觀呈現耦合效應造成的額外厚度損失。
- **可展開表格：`Galvanic Series Overview (Select Material Subset)`**
  - 可勾選多種材料，檢視它們在當前資料庫中的 <i>E</i><sub>corr</sub> 中位數排序（由貴到賤），協助在設計階段避開電位差過大的材料組合。

#### 分頁 3：`Recipe Comparison`（多組清洗配方比較）
可透過滑桿設定 **2 至 5 組**候選清洗配方（預設提供 5 種常見情境作為範本，皆可自由修改藥液濃度、pH、溫度與時間），系統會自動計算並於上方綠色橫幅顯示 **Lowest-Risk Recommendation**（風險最低的推薦配方）。

- **表格：`Side-by-Side Cleaning Recipe Rankings`**
  - 依綜合風險由低到高自動排名（`Rank 1` 為最佳），並排呈現每個配方的化學組成、初始腐蝕速率、累積深度、達門檻時間、風險分數／等級，以及（若為雙金屬模式）耦合加速倍率。
- **互動圖表**：
  1. **關鍵指標長條圖（`Cleaning Recipe Comparison — Key Metrics`）**：以分組長條圖並排比較各配方的腐蝕深度與風險分數，系統推薦的最佳配方會以綠色醒目標示。
  2. **正規化安全雷達圖（`Normalized Safety Radar`）**：將多項指標正規化到同一尺度後繪成雷達圖；**圖形涵蓋面積越大、越靠外側的配方，代表綜合安全性越高／腐蝕風險越低**。

#### 分頁 4：`Report Export`（多格式技術報告匯出）
將目前所有模擬結果（含輸入條件、參數來源、表格、圖表、不確定性區間、信心等級與配方排名）打包匯出為 6 種格式：

| 匯出格式 | 檔名副檔名 | 適用場景 |
| :--- | :--- | :--- |
| HTML 完整報告 | `.html` | 瀏覽器開啟檢視完整排版、互動圖表與摘要表格 |
| PDF 技術報告 | `.pdf` | 適合正式文件存檔、實驗室列印或團隊傳閱 |
| JSON 結構化資料 | `.json` | 供程式化後處理、自動化管線或資料庫串接 |
| CSV 彙整表 | `.csv` | 使用 Excel 或 pandas 快速開啟進行二次統計 |
| PNG 高解析複合圖 | `.png` | 直接嵌入簡報（PPT）或報告文件的點陣圖 |
| SVG 向量圖表 | `.svg` | 適合論文發表或高倍率放大不失真的向量圖 |

頁面下方同步提供 **Preview Composite Report Figure**，可直接預覽匯出的複合圖表內容。

#### 常見介面專有名詞速查

| 名詞 / 符號 | 中文說明 | 判讀重點 |
| :--- | :--- | :--- |
| <i>E</i><sub>corr</sub> | 自腐蝕電位（V vs. SHE） | 數值越低（越負）代表熱力學上越活潑、越容易在雙金屬耦合中成為陽極 |
| <i>i</i><sub>corr</sub> | 自腐蝕電流密度（µA/cm²） | 正比於瞬時腐蝕速率，數值越大代表腐蝕越快 |
| <i>β</i><sub>a</sub>, <i>β</i><sub>c</sub> | 陽極 / 陰極 Tafel 斜率（mV/dec） | 描述電位每改變一個數量級電流所需之電壓變化，影響極化曲線斜率 |
| CR<sub>0</sub> | 初始腐蝕速率（nm/min、mm/yr） | 由 <i>i</i><sub>corr</sub> 透過 Faraday 定律換算的薄膜厚度損失速率 |
| <i>E</i><sub>galv</sub>, <i>i</i><sub>galv</sub> | 伽凡尼混合電位與耦合電流密度 | 雙金屬接觸後達成電流守恆的新平衡點 |
| Acceleration Factor | 伽凡尼加速倍率 | `i_galv / i_corr,active`；大於 `1.0×` 表示耦合加劇了主動金屬腐蝕 |
| <i>A</i><sub>noble</sub> / <i>A</i><sub>active</sub> | 陰陽極面積比 | 貴金屬（陰極）面積相對越大，對主動金屬（陽極）的加速效應越強烈 |
| <i>d</i><sub>crit</sub> | 臨界厚度門檻（nm） | 製程可容許的最大腐蝕厚度損失上限 |
| <i>d</i><sub>max</sub> | 鈍化極限深度（nm） | Passivation 動力學模型中，表面生成鈍化膜後腐蝕深度趨近的飽和上限 |
| Time to Threshold | 達門檻時間 | 腐蝕深度達到 <i>d</i><sub>crit</sub> 所需的浸泡秒數，越長越安全 |
| Confidence Tier | 適用域信心等級（High / Medium / Low） | 評估查詢條件是否落在 ML 訓練資料覆蓋範圍內；`Low` 代表外推風險高，建議補做實驗 |

### 步驟 4：匯入自己的實驗資料（選用）
將新的電化學、材料或溶液 CSV 檔案放進 `data/raw/`（可依金屬分資料夾存放），再執行一次 `build_artifacts` 指令重建 `data/processed/`，重新整理網頁即可載入新資料與新模型。

---

## 五、常見問題與排錯指南

| 常見現象 / 錯誤訊息 | 原因說明 | 解決方法 |
| :--- | :--- | :--- |
| 左側邊欄顯示 `Raw CSV files in data/raw/ have changed...` 或 `Prebuilt artifacts not found` | `data/raw/` 檔案已變動但尚未重建快取，或 `data/processed/` 缺少預建檔。 | 執行 `build_artifacts` 指令重新建置預建資料表與模型，完成後重新整理瀏覽器。 |
| Windows PowerShell 執行 `Activate.ps1` 出現 `running scripts is disabled` | Windows 預設禁止執行未簽署的 PowerShell 腳本。 | **最簡單做法**：不需啟用虛擬環境，直接使用本文件表格中的 `.venv\Scripts\python.exe` 與 `.venv\Scripts\streamlit.exe` 即可。<br>若仍想啟用，請先執行：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。 |
| 啟動時出現 `TypeError: ... got an unexpected keyword argument 'width'` | 安裝的 `streamlit` 版本低於 `1.50.0`。 | 升級套件：`pip install --upgrade -r requirements.txt`（確認 `streamlit >= 1.50.0`）。 |
| 報錯 `ImportError: Unable to find a usable engine` 或 Parquet 讀取失敗 | 缺少 `pyarrow` 套件。 | 執行 `pip install "pyarrow>=16.0.0"` 或重新安裝 `requirements.txt`。 |
| 啟動時出現 `Address already in use` 或 `Port 8501 is already in use` | 已有另一個 Streamlit 實例或程式佔用 `8501` 埠。 | 關閉舊的終端機視窗，或改用 `--server.port 8502` 指定新埠號啟動。 |
| 材料下拉選單標示 `no Faraday props` 或 CR<sub>0</sub> 顯示 `N/A` | 該材料缺少密度（density）或當量重（equivalent weight），無法進行 Faraday 速率換算。 | 在 `data/raw/material_properties.csv` 補齊該材料的物理性質，然後重新執行 `build_artifacts`。 |
| 首次切換至 `Gaussian Process` 或 `RF + GP Hybrid` 時等待較久 | 預設離線預建僅包含 3 個樹模型，其餘模型會在首次選取時即時訓練並快取。 | 若常使用這些模型，可先執行 `build_artifacts --models all` 離線預建。 |
| 畫面上方出現紅色 `Confidence Tier: Low` 警告 | 查詢條件超出訓練資料的適用域（Applicability Domain），或該材料樣本數過少（`n < 10`）。 | 展開 **Applicability Domain Warnings** 與 **Suggested Follow-Up Experiments** 查看建議補強的實驗條件，或切換至 `Manual Override` 輸入實測數據。 |
| Windows 執行 `python` 跳出 Microsoft Store | 系統未安裝 Python 或未加入 PATH。 | 至 [python.org](https://www.python.org/downloads/) 安裝 Python 3.11+（安裝時務必勾選 **Add python.exe to PATH**），並改用 `py -3` 執行。 |

---

## 六、示範 Notebook 與進階文件

- **五份方法論示範 Notebook（`notebooks/`）**：
  1. `01_data_exploration_and_cleaning.ipynb` — 原始資料清理、單位／參考電極標準化與伽凡尼序列建立。
  2. `02_phase1_tafel_and_corrosion_calculations.ipynb` — Wagner–Traud 極化曲線、Faraday 速率計算與驅動力篩選。
  3. `03_phase2_ml_training_and_uncertainty.ipynb` — 特徵工程、跨模型交叉驗證、Monte Carlo 不確定性與適用域分析。
  4. `04_bimetallic_galvanic_mixed_potential.ipynb` — 面積加權混合電位求解、面積比掃描與疊加極化圖。
  5. `05_time_resolved_kinetics_and_risk.ipynb` — Linear / Passivation / Power-Law 腐蝕深度軌跡與風險曲線。
- **領域術語表**：[GLOSSARY.md](GLOSSARY.md)
- **架構決策紀錄（ADRs）**：`docs/adr/0001` 至 `docs/adr/0003`
