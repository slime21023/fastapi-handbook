# FastAPI 中的依賴注入

## 1. 基本依賴注入（Basic Injection）

基本依賴注入（Basic Injection）是依賴注入的核心功能，通過將依賴的管理交由框架處理，實現模塊間的低耦合設計。它適用於簡單的依賴場景，例如配置參數的傳遞或靜態資源的初始化。

**FastAPI 中的基本依賴注入**: 依賴注入通過 `Depends` 函數實現。開發者可以定義一個函數作為依賴，並在路由處通過 `Depends` 引入該依賴。

````python
from fastapi import FastAPI, Depends

app = FastAPI()

# 定義依賴函數
def get_config():
    return {"app_name": "MyApp", "version": "1.0"}

# 使用依賴
@app.get("/")
def read_root(config: dict = Depends(get_config)):
    return {"message": f"Welcome to {config['app_name']} v{config['version']}"}
````

**關鍵特性**
- **簡單易用**：只需定義依賴函數並通過 `Depends` 引入。
- **提升代碼可維護性**：依賴的定義與使用分離，讓邏輯更清晰。
- **支持默認值與可選依賴**：依賴函數可以設置默認值，避免因缺少依賴而導致錯誤。

**常見應用場景**
- **配置管理**：將應用配置集中管理並注入到需要的模塊中。
- **用戶上下文**：提取當前用戶信息並傳遞給處理邏輯。
- **靜態資源初始化**：如紀錄日誌對象或靜態文件路徑。

---

## 2. 多層依賴注入（Multi-layer Injection）


多層依賴注入（Multi-layer Injection）是依賴注入的一種進階形式，適用於依賴之間存在層級關係的場景。這種設計模式允許開發者將複雜的依賴邏輯分層處理，從而提升系統的靈活性與可維護性。

**FastAPI 中的多層依賴注入**: 在 FastAPI 中，依賴可以相互嵌套，實現多層的依賴關係。每一層依賴都可以通過 `Depends` 引入其他依賴。

````python
from fastapi import FastAPI, Depends

app = FastAPI()

# 第一層依賴：獲取配置
def get_config():
    return {"app_name": "MyApp", "version": "1.0"}

# 第二層依賴：基於配置生成服務
def get_service(config: dict = Depends(get_config)):
    return {"service_name": f"{config['app_name']} Service"}

# 使用多層依賴
@app.get("/")
def read_root(service: dict = Depends(get_service)):
    return {"message": f"Welcome to {service['service_name']}"}
````

### 關鍵特性
- **層次分明**：依賴之間的層級關係清晰，便於管理。
- **靈活組合**：可以根據需求動態組合多層依賴。
- **提升測試效率**：每一層依賴都可以單獨測試，降低測試的複雜度。

### 常見應用場景
- **資源分層管理**：如基於配置生成數據庫連接池或其他服務對象。
- **依賴的動態構建**：根據業務需求，動態組合多層依賴。
- **複雜業務邏輯的分層處理**：例如，將數據驗證與數據處理分層。

---

## 3. 異步依賴注入（Async Injection）


異步依賴注入（Async Injection）專為需要異步操作的場景設計，例如網絡請求、數據庫查詢或文件操作。FastAPI 原生支持異步依賴，允許開發者在依賴函數中使用 `async` 關鍵字來處理異步邏輯。

**FastAPI 中的異步依賴注入**: FastAPI 的依賴注入機制完全兼容異步函數，這使得處理高併發場景更加高效。

````python
from fastapi import FastAPI, Depends

app = FastAPI()

# 定義異步依賴
async def fetch_data():
    # 模擬異步操作，例如從遠程 API 獲取數據
    return {"data": "Sample data"}

# 使用異步依賴
@app.get("/")
async def read_root(data: dict = Depends(fetch_data)):
    return {"message": f"Fetched data: {data['data']}"}
````

**關鍵特性**
- **高性能**：利用 Python 的異步特性，支持高併發場景。
- **適配異步資源**：方便集成異步數據庫、外部 API 或其他異步操作。
- **與同步依賴一致的接口**：開發者無需學習額外的語法，異步依賴的使用方式與同步依賴完全一致。

**常見應用場景**
- **外部 API 數據拉取**：如從第三方服務獲取數據。
- **異步數據庫操作**：如使用異步 ORM（例如 Tortoise ORM 或 SQLAlchemy Async）。
- **文件讀寫或網絡請求**：處理大文件或高頻請求場景。

