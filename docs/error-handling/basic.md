# FastAPI 異常處理基礎

FastAPI 是一個現代化的 Python Web 框架，它基於 Starlette 和 Pydantic，提供了強大的異常處理機制。在 FastAPI 中，異常處理是確保 API 穩定性和提供良好用戶體驗的關鍵部分。

## 1.1 異常處理的重要性

在 API 開發中，異常處理至關重要，因為它能夠帶來以下優勢：

| 優勢 | 說明 |
|------|------|
| **提供清晰的錯誤信息** | 向客戶端提供明確、結構化的錯誤信息，幫助開發者快速理解問題所在 |
| **防止應用崩潰** | 捕獲並妥善處理異常，確保應用在面對錯誤時仍能保持穩定運行 |
| **維護一致的 API 響應格式** | 確保所有錯誤響應遵循統一的格式，便於客戶端處理 |
| **提高安全性** | 避免在錯誤信息中洩露敏感信息，同時提供足夠的信息幫助調試 |
| **改善用戶體驗** | 通過友好的錯誤信息，幫助 API 消費者更好地理解和解決問題 |
| **簡化調試和監控** | 標準化的錯誤處理使得日誌記錄和監控更加容易 |

## 1.2 FastAPI 異常處理機制

FastAPI 的異常處理機制建立在 Starlette 的基礎上，並進行了擴展，提供了更加豐富和靈活的功能。

### 基本異常處理流程

當 FastAPI 應用中發生異常時，處理流程如下：

1. 異常被拋出（顯式或隱式）
2. FastAPI 捕獲異常
3. 查找對應的異常處理器
4. 將異常轉換為適當的 HTTP 響應
5. 返回包含錯誤詳情的 JSON 響應給客戶端

### HTTPException

FastAPI 提供了 `HTTPException` 類，這是處理 HTTP 錯誤最基本的方式：

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id}
```

在上面的例子中，當 `item_id` 為 0 時，會拋出一個 404 錯誤，客戶端將收到如下 JSON 響應：

```json
{
  "detail": "Item not found"
}
```

### HTTPException 參數

`HTTPException` 支持以下參數：

| 參數 | 類型 | 說明 |
|------|------|------|
| `status_code` | int | HTTP 狀態碼 |
| `detail` | Any | 錯誤詳情，將作為響應的 `detail` 字段 |
| `headers` | dict | 可選，要包含在響應中的額外 HTTP 標頭 |

### 異常處理器

FastAPI 允許您註冊自定義的異常處理器，用於捕獲和處理特定類型的異常：

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

class CustomException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something wrong."}
    )

@app.get("/custom-error")
async def trigger_custom_error():
    raise CustomException(name="Someone")
```

當訪問 `/custom-error` 端點時，客戶端將收到一個 418 狀態碼和自定義的錯誤信息。

### 請求驗證異常

FastAPI 使用 Pydantic 進行請求數據驗證。當請求數據不符合模型定義時，會自動拋出 `RequestValidationError`：

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float = Field(..., gt=0)
    is_offer: bool = False

@app.post("/items/")
async def create_item(item: Item):
    return item
```

如果客戶端發送的請求中 `price` 小於或等於 0，FastAPI 會自動返回 422 狀態碼和詳細的驗證錯誤信息。

### 默認異常處理

FastAPI 為常見的錯誤情況提供了默認的異常處理器：

| 異常類型 | 說明 | 默認狀態碼 |
|---------|------|-----------|
| **RequestValidationError** | 請求數據不符合 Pydantic 模型定義 | 422 |
| **路徑參數轉換錯誤** | 路徑參數無法轉換為指定類型 | 422 |
| **HTTPException** | 顯式拋出的 HTTP 異常 | 由異常指定 |

### 異常處理的優先級

FastAPI 按照以下優先級處理異常：

| 優先級 | 處理器類型 |
|-------|-----------|
| 1 | 為特定異常類型註冊的自定義異常處理器 |
| 2 | 為異常基類註冊的異常處理器 |
| 3 | 默認的異常處理器 |

### 常見異常與狀態碼對應表

以下是一些常見的 HTTP 狀態碼及其適用場景：

| 狀態碼 | 名稱 | 適用場景 |
|-------|------|---------|
| 400 | Bad Request | 客戶端請求有誤，如參數格式錯誤 |
| 401 | Unauthorized | 未提供身份驗證或身份驗證失敗 |
| 403 | Forbidden | 已認證但無權訪問資源 |
| 404 | Not Found | 請求的資源不存在 |
| 405 | Method Not Allowed | 不支持該 HTTP 方法 |
| 409 | Conflict | 請求與服務器當前狀態衝突 |
| 422 | Unprocessable Entity | 請求格式正確但語義錯誤 |
| 429 | Too Many Requests | 請求頻率超過限制 |
| 500 | Internal Server Error | 服務器內部錯誤 |
| 503 | Service Unavailable | 服務暫時不可用 |

### 簡單示例

下面是一個結合了多種異常處理機制的簡單示例：

```python
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI()

# 模型定義
class Item(BaseModel):
    name: str
    price: float = Field(..., gt=0)

# 自定義異常
class OutOfStockError(Exception):
    def __init__(self, item_name: str):
        self.item_name = item_name

# 處理驗證錯誤
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": exc.errors()
        }
    )

# 處理自定義異常
@app.exception_handler(OutOfStockError)
async def out_of_stock_exception_handler(request: Request, exc: OutOfStockError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "message": f"The item '{exc.item_name}' is out of stock"
        }
    )

# API 端點
@app.post("/items/")
async def create_item(item: Item):
    if item.name == "banana":
        raise OutOfStockError(item_name=item.name)
    
    if item.name == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot use 'error' as an item name"
        )
    
    return {"status": "success", "item": item.dict()}
```

### 異常處理響應示例

下表展示了不同情況下的請求和響應示例：

| 請求 | 情況 | 響應狀態碼 | 響應內容 |
|------|------|-----------|---------|
| `POST /items/` 請求體: `{"name": "apple", "price": 5.99}` | 有效請求 | 200 | `{"status": "success", "item": {"name": "apple", "price": 5.99}}` |
| `POST /items/` 請求體: `{"name": "apple", "price": -5.99}` | 驗證錯誤 | 422 | `{"status": "error", "message": "Validation error", "errors": [...]}` |
| `POST /items/` 請求體: `{"name": "banana", "price": 3.99}` | 自定義異常 | 400 | `{"status": "error", "message": "The item 'banana' is out of stock"}` |
| `POST /items/` 請求體: `{"name": "error", "price": 1.99}` | HTTP 異常 | 400 | `{"detail": "You cannot use 'error' as an item name"}` |

## 小結

FastAPI 提供了強大而靈活的異常處理機制，使開發者能夠:

- 使用 `HTTPException` 處理基本的 HTTP 錯誤
- 創建自定義異常類型處理特定的業務邏輯錯誤
- 註冊異常處理器來自定義錯誤響應格式
- 利用 Pydantic 進行自動的請求數據驗證

掌握這些基礎知識，將幫助您構建更加健壯、用戶友好的 FastAPI 應用。在接下來的章節中，我們將深入探討更多高級的異常處理技術和最佳實踐。