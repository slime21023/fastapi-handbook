# OpenAPI 與 FastAPI 故障排除指南

## 1. OpenAPI 文檔生成問題

### 1.1 Swagger UI 顯示問題

| 問題 | 可能原因 | 解決方案 |
|------|---------|---------|
| Swagger UI 完全無法載入 | 路由定義錯誤或模型定義問題 | 檢查控制台錯誤信息；啟用詳細日誌 [2] |
| API 文檔內容不完整 | OpenAPI 生成錯誤 | 暫時禁用 OpenAPI 並逐步啟用功能來定位問題 [1] |
| 示例值顯示不正確 | 模型配置問題 | 使用 `Field` 參數正確設置示例值 [3] |

```python
# 啟用詳細日誌進行調試
import logging
logging.basicConfig(level=logging.DEBUG)

# 或使用 uvicorn 的詳細日誌
# uvicorn main:app --log-level debug

# 暫時禁用 OpenAPI 以定位問題
app = FastAPI(openapi_url=None)
```
[1]

### 1.2 自定義 OpenAPI 文檔

| 問題 | 解決方案 | 示例 |
|------|---------|-----|
| 標籤順序混亂 | 自定義 OpenAPI 模式 | 使用 `get_openapi()` 並修改標籤順序 |
| 描述信息缺失 | 正確設置路由和模型描述 | 在路由和模型定義中添加 docstring |
| 複雜模型序列化失敗 | 簡化模型結構 | 拆分複雜模型為更小的組件 [1] |

```python
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="自定義 API 文檔",
        version="1.0.0",
        description="這是一個自定義的 API 文檔",
        routes=app.routes,
    )
    
    # 自定義標籤順序
    openapi_schema["tags"] = [
        {"name": "users", "description": "用戶操作"},
        {"name": "items", "description": "項目操作"},
        # 其他標籤...
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```
[1]

## 2. 路由和請求處理問題

### 2.1 路徑參數問題

| 問題 | 症狀 | 解決方案 |
|------|------|---------|
| 路徑順序衝突 | 特定路由無法訪問 | 調整路由順序，固定路徑先於參數路徑 [4] |
| 參數類型轉換失敗 | 404 錯誤 | 檢查參數類型定義；添加路徑參數驗證 |
| 路徑參數解析錯誤 | 500 內部服務器錯誤 | 使用 `Path()` 添加額外驗證 [4] |

```python
# 正確的路由順序
@app.get("/users/me")  # 固定路徑先定義
async def read_current_user():
    return {"user_id": "current"}

@app.get("/users/{user_id}")  # 參數路徑後定義
async def read_user(user_id: int = Path(..., ge=1)):
    return {"user_id": user_id}
```
[4]

### 2.2 請求體解析問題

| 問題 | 錯誤信息 | 調試方法 |
|------|---------|---------|
| JSON 解析失敗 | `json.decoder.JSONDecodeError` | 自定義異常處理器捕獲並記錄原始請求體 [4] |
| 表單數據處理錯誤 | `FormException` | 檢查表單字段名稱和類型；驗證 `enctype` 設置 |
| 文件上傳問題 | `UploadFileException` | 檢查文件大小限制；驗證 `multipart/form-data` 設置 [3] |

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 獲取並記錄原始請求體
    body = await request.body()
    logger.error(f"請求驗證錯誤: {exc.errors()}")
    logger.debug(f"原始請求體: {body.decode()}")
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )
```
[4]

## 3. 數據驗證和模型問題

### 3.1 Pydantic 模型驗證錯誤

| 錯誤類型 | 常見原因 | 解決方案 |
|---------|---------|---------|
| 字段類型不匹配 | 前端發送的數據類型錯誤 | 添加詳細的錯誤信息；使用 `field_validator` [1] |
| 缺少必填字段 | 請求中缺少必要字段 | 檢查請求數據；使用 `Field(...)` 標記必填字段 |
| 自定義驗證失敗 | 驗證邏輯錯誤 | 添加調試日誌；使用 `debug_validation` 方法 [1] |

```python
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=8)
    
    @field_validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("無效的電子郵件格式，必須包含 @ 符號")
        return v
    
    # 調試模型驗證
    @classmethod
    def debug_validation(cls, data: dict):
        try:
            return cls(**data)
        except ValidationError as e:
            print(f"驗證錯誤: {e}")
            print(f"錯誤詳情: {e.errors()}")
            print(f"輸入數據: {data}")
            raise
```
[1]

### 3.2 響應模型問題

| 問題 | 症狀 | 解決方案 |
|------|------|---------|
| 響應序列化失敗 | 500 內部服務器錯誤 | 檢查返回數據結構；簡化響應模型 [2] |
| 循環引用 | OpenAPI 生成錯誤 | 使用 `Optional` 和延遲引用；拆分模型 [1] |
| 模型嵌套過深 | 性能問題或序列化錯誤 | 減少嵌套層級；使用更扁平的結構 [3] |

```python
# 處理循環引用問題
class Parent(BaseModel):
    name: str
    # 使用 Optional 和延遲引用
    children: Optional[List["Child"]] = None

class Child(BaseModel):
    name: str
    parent_name: str  # 只引用父級的名稱而不是整個對象

# 更新 Pydantic 模型配置
Parent.model_rebuild()

# 響應序列化錯誤調試
@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int):
    try:
        result = get_item_from_db(item_id)
        return result
    except Exception as e:
        logger.error(f"響應序列化錯誤: {e}")
        # 返回簡化的響應以便調試
        return {"id": item_id, "name": "調試項目"}
```
[1] [3]

## 4. 認證和安全問題

### 4.1 OAuth2 和 JWT 問題

| 問題 | 錯誤信息 | 調試方法 |
|------|---------|---------|
| 令牌驗證失敗 | `Could not validate credentials` | 檢查密鑰和算法；記錄詳細的 JWT 錯誤 [2] |
| 令牌過期 | `Token expired` | 檢查令牌過期時間；添加令牌刷新機制 |
| 範圍權限錯誤 | `Not enough permissions` | 記錄令牌範圍和端點所需範圍 [2] |

```python
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        # 記錄令牌信息（不記錄完整令牌）
        logger.debug(f"驗證令牌: {token[:10]}...")
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            logger.warning("令牌缺少 'sub' 字段")
            raise credentials_exception
            
        # 檢查令牌是否過期
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            logger.warning(f"令牌已過期: {datetime.fromtimestamp(exp)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return {"username": username}
    except JWTError as e:
        logger.error(f"JWT 錯誤: {str(e)}")
        raise credentials_exception
```
[2]

### 4.2 CORS 配置問題

| 問題 | 瀏覽器錯誤信息 | 解決方案 |
|------|--------------|---------|
| 前端無法訪問 API | `Access-Control-Allow-Origin` 錯誤 | 正確配置 `allow_origins`；添加所有必要的源 [3] |
| 預檢請求失敗 | `Method OPTIONS is not allowed` | 確保正確處理 OPTIONS 請求；配置 `allow_methods` |
| 憑證請求被拒絕 | `Credentials is not supported` | 設置 `allow_credentials=True` [3] |

```python
# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://frontend.example.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CORS 調試中間件
@app.middleware("http")
async def log_cors_requests(request, call_next):
    if request.method == "OPTIONS":
        logger.debug("收到 OPTIONS 預檢請求")
        logger.debug(f"Origin: {request.headers.get('origin')}")
    
    response = await call_next(request)
    
    if "origin" in request.headers:
        logger.debug(f"CORS 響應頭:")
        logger.debug(f"Access-Control-Allow-Origin: {response.headers.get('access-control-allow-origin')}")
    
    return response
```
[3]

## 5. 性能和部署問題

### 5.1 API 響應性能問題

| 問題 | 症狀 | 診斷方法 |
|------|------|---------|
| 慢速數據庫查詢 | API 響應延遲 | 添加性能監控中間件；記錄查詢執行時間 [2] |
| 阻塞操作 | 服務器吞吐量下降 | 使用異步操作；避免在事件循環中執行阻塞代碼 |
| 資源限制 | 服務器 CPU/內存使用率高 | 監控系統資源；優化資源使用 [2] |

```python
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # 記錄處理時間
    response.headers["X-Process-Time"] = str(process_time)
    
    # 對於慢請求進行警告
    if process_time > 1.0:  # 超過 1 秒的請求
        logger.warning(
            f"慢請求: {request.method} {request.url.path} 耗時 {process_time:.4f}s"
        )
        # 添加更多診斷信息
        logger.debug(f"請求參數: {request.query_params}")
    
    return response
```
[2]

### 5.2 部署和環境問題

| 問題 | 可能原因 | 解決方案 |
|------|---------|---------|
| 生產環境與開發環境不一致 | 環境配置差異 | 使用環境變量；實現環境特定配置 [2] |
| 依賴項衝突 | 包版本不兼容 | 使用虛擬環境；固定依賴項版本 |
| 代理和負載均衡問題 | 代理配置不正確 | 設置正確的主機頭；配置 `root_path` [3] |

```python
# 使用環境變量進行配置
import os
from fastapi import FastAPI
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "FastAPI App"
    debug: bool = False
    database_url: str
    
    class Config:
        env_file = ".env"

settings = Settings()
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    # 如果在代理後面運行，設置 root_path
    root_path=os.getenv("ROOT_PATH", "")
)
```
[2] [3]

## 6. 調試技巧與工具

### 6.1 日誌配置與分析

| 日誌級別 | 適用場景 | 配置示例 |
|---------|---------|---------|
| DEBUG | 開發環境；詳細調試 | 記錄請求/響應詳情；包含堆棧跟踪 |
| INFO | 一般操作信息 | 記錄請求開始/結束；基本流程信息 |
| WARNING | 潛在問題 | 慢響應；資源使用率高 |
| ERROR | 錯誤但可恢復 | 請求處理失敗；數據庫連接問題 |
| CRITICAL | 嚴重錯誤 | 應用崩潰；數據損壞 [2] |

```python
# 結構化日誌配置
import logging
import uuid
from fastapi import FastAPI, Request

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(request_id)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)

logger = logging.getLogger("app")

# 添加請求 ID 中間件
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    logger.info(f"Request started: {request.method} {request.url.path}", 
                extra={"request_id": request_id})
    
    try:
        response = await call_next(request)
        logger.info(f"Request completed: {response.status_code}", 
                    extra={"request_id": request_id})
        return response
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", 
                     extra={"request_id": request_id}, exc_info=True)
        raise
```
[2]

### 6.2 調試工具與技巧

| 工具 | 用途 | 使用場景 |
|------|------|---------|
| FastAPI 測試客戶端 | API 端點測試 | 單元測試；本地調試 [3] |
| Pydantic 調試模式 | 數據模型驗證 | 檢查模型驗證錯誤 |
| Python 調試器 (pdb) | 代碼執行調試 | 複雜邏輯調試；條件斷點 [3] |
| API 文檔 UI | 交互式 API 測試 | 快速測試端點；檢查請求/響應模式 [2] |

```python
# 使用 FastAPI 測試客戶端進行調試
from fastapi.testclient import TestClient

client = TestClient(app)

def debug_api():
    # 發送測試請求
    response = client.get("/items/42", headers={"X-Test": "test"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # 檢查請求頭和響應頭
    print(f"Request Headers: {response.request.headers}")
    print(f"Response Headers: {response.headers}")
    
    # 如果有錯誤，打印錯誤信息
    if response.status_code >= 400:
        print(f"Error: {response.text}")
```
[3]

## 7. 常見問題快速參考表

| 問題類別 | 常見症狀 | 首要檢查項 | 參考章節 |
|---------|---------|-----------|---------|
| Swagger UI 無法載入 | `/docs` 頁面空白或報錯 | 控制台錯誤；路由定義 | 1.1 |
| 路由訪問 404 | 無法訪問特定端點 | 路由順序；路徑參數定義 | 2.1 |
| 請求驗證錯誤 | 422 Unprocessable Entity | 請求數據格式；模型驗證規則 | 3.1 |
| 認證失敗 | 401 Unauthorized | 令牌有效性；認證配置 | 4.1 |
| CORS 錯誤 | 瀏覽器控制台跨域錯誤 | CORS 中間件配置；預檢請求處理 | 4.2 |
| 響應緩慢 | API 請求延遲高 | 數據庫查詢；阻塞操作 | 5.1 |
| 部署問題 | 生產環境異常 | 環境配置；代理設置 | 5.2 |
| OpenAPI 生成錯誤 | 文檔生成失敗 | 模型循環引用；不支持的類型 | 1.2 [1] [2] [3] [4] |

## 8. 總結

FastAPI 和 OpenAPI 故障排除的關鍵步驟：

1. **識別問題類型**：確定問題屬於哪個類別（文檔生成、路由處理、數據驗證、認證等）
2. **啟用詳細日誌**：設置適當的日誌級別，捕獲詳細的錯誤信息
3. **隔離問題**：通過禁用或簡化相關功能來定位問題根源
4. **使用適當工具**：根據問題類型選擇合適的調試工具和技術
5. **查閱文檔**：參考 FastAPI 官方文檔中的相關指南和最佳實踐 [4]

通過系統性地應用這些故障排除技巧，開發者可以更高效地解決 FastAPI 和 OpenAPI 開發中遇到的各種問題，構建更加穩健和可靠的 API 服務。

[1]: https://stackoverflow.com/questions/70257170/how-to-debug-fastapi-openapi-generation-error
[2]: https://www.restack.io/p/fastapi-answer-swagger-not-working
[3]: https://www.restack.io/p/fastapi-answer-examples-not-working
[4]: https://fastapi.tiangolo.com/tutorial/handling-errors/