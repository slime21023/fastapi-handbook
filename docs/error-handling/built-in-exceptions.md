# 2. 內建異常類型與 HTTP 狀態碼

FastAPI 提供了多種內建的異常類型和 HTTP 狀態碼，用於處理各種錯誤情況。了解這些內建機制對於構建健壯的 API 至關重要。

## 2.1 HTTPException

`HTTPException` 是 FastAPI 中最常用的異常類型，它允許您指定 HTTP 狀態碼和錯誤詳情，並可選地添加自定義標頭。

### 基本用法

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id < 0:
        raise HTTPException(status_code=400, detail="Item ID must be positive")
    if item_id == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "name": f"Item {item_id}"}
```

### HTTPException 參數詳解

| 參數 | 類型 | 必需 | 說明 |
|------|------|------|------|
| `status_code` | int | 是 | HTTP 狀態碼 |
| `detail` | Any | 否 | 錯誤詳情，將作為響應的 `detail` 字段 |
| `headers` | dict | 否 | 要包含在響應中的額外 HTTP 標頭 |

### 響應格式

當拋出 `HTTPException` 時，FastAPI 會自動生成一個 JSON 響應，格式如下：

```json
{
  "detail": "錯誤信息"
}
```

## 2.2 常見 HTTP 狀態碼

HTTP 狀態碼分為五類，每類用於表示不同類型的響應：

| 類別 | 範圍 | 說明 |
|------|------|------|
| 1xx | 100-199 | 信息性狀態碼，表示請求已被接收並正在處理 |
| 2xx | 200-299 | 成功狀態碼，表示請求已成功被接收、理解和處理 |
| 3xx | 300-399 | 重定向狀態碼，表示需要客戶端進一步操作才能完成請求 |
| 4xx | 400-499 | 客戶端錯誤狀態碼，表示客戶端請求包含錯誤或無法被服務器處理 |
| 5xx | 500-599 | 服務器錯誤狀態碼，表示服務器在處理請求時發生錯誤 |

### 常用 HTTP 狀態碼詳解

以下是 API 開發中最常用的 HTTP 狀態碼：

| 狀態碼 | 名稱 | 說明 | 使用場景 |
|-------|------|------|---------|
| 200 | OK | 請求成功 | 成功獲取資源、成功更新資源 |
| 201 | Created | 資源創建成功 | 成功創建新資源 |
| 204 | No Content | 請求成功，但無返回內容 | 成功刪除資源 |
| 400 | Bad Request | 客戶端請求有誤 | 請求參數格式錯誤、邏輯錯誤 |
| 401 | Unauthorized | 未提供身份驗證或身份驗證失敗 | 缺少認證信息、認證信息無效 |
| 403 | Forbidden | 已認證但無權訪問資源 | 用戶無權執行操作 |
| 404 | Not Found | 請求的資源不存在 | 資源 ID 不存在 |
| 405 | Method Not Allowed | 不支持該 HTTP 方法 | 嘗試使用不支持的 HTTP 方法 |
| 409 | Conflict | 請求與服務器當前狀態衝突 | 資源已存在、版本衝突 |
| 422 | Unprocessable Entity | 請求格式正確但語義錯誤 | 請求數據驗證失敗 |
| 429 | Too Many Requests | 請求頻率超過限制 | 超過 API 速率限制 |
| 500 | Internal Server Error | 服務器內部錯誤 | 未捕獲的異常、系統錯誤 |
| 503 | Service Unavailable | 服務暫時不可用 | 服務器過載、維護中 |

### 在 FastAPI 中使用狀態碼

FastAPI 提供了 `status` 模塊，包含所有標準 HTTP 狀態碼的常量：

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Item ID must be positive"
        )
    if item_id == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found"
        )
    return {"item_id": item_id, "name": f"Item {item_id}"}
```

使用 `status` 模塊的常量比直接使用數字狀態碼更具可讀性和可維護性。

### 常見狀態碼使用示例

| 狀態碼常量 | 使用示例 |
|-----------|---------|
| `status.HTTP_400_BAD_REQUEST` | `raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid parameter")` |
| `status.HTTP_401_UNAUTHORIZED` | `raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")` |
| `status.HTTP_403_FORBIDDEN` | `raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")` |
| `status.HTTP_404_NOT_FOUND` | `raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")` |
| `status.HTTP_409_CONFLICT` | `raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")` |
| `status.HTTP_422_UNPROCESSABLE_ENTITY` | `raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid email format")` |
| `status.HTTP_500_INTERNAL_SERVER_ERROR` | `raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error")` |

## 2.3 帶有自定義標頭的異常

有時您需要在錯誤響應中包含額外的 HTTP 標頭，例如在身份驗證失敗時指示客戶端如何進行身份驗證：

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

@app.get("/protected-resource/")
async def get_protected_resource(token: str = None):
    if token != "valid_token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"message": "You have access to the protected resource"}
```

在上面的例子中，當認證失敗時，響應中會包含 `WWW-Authenticate: Bearer` 標頭，告訴客戶端應該使用 Bearer 令牌進行身份驗證。

### 常見的自定義標頭場景

| 場景 | 標頭 | 示例 |
|------|------|------|
| 身份驗證失敗 | `WWW-Authenticate` | `{"WWW-Authenticate": "Bearer"}` |
| 重定向 | `Location` | `{"Location": "/new-url"}` |
| 速率限制 | `Retry-After` | `{"Retry-After": "60"}` |
| 跨域資源共享 | `Access-Control-Allow-Origin` | `{"Access-Control-Allow-Origin": "*"}` |

## 2.4 FastAPI 內建的其他異常類型

除了 `HTTPException`，FastAPI 還提供了其他幾種內建異常類型：

### RequestValidationError

當請求數據不符合 Pydantic 模型定義時，FastAPI 會自動拋出 `RequestValidationError`：

```python
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float = Field(..., gt=0)

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

@app.post("/items/")
async def create_item(item: Item):
    return item
```

### WebSocketException

用於 WebSocket 連接中的異常：

```python
from fastapi import FastAPI, WebSocket, WebSocketException, status

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data == "error":
                raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
            await websocket.send_text(f"Message received: {data}")
    except WebSocketException:
        await websocket.close()
```

### StarletteHTTPException

FastAPI 的 `HTTPException` 實際上是基於 Starlette 的 `HTTPException`。在大多數情況下，您應該使用 FastAPI 的 `HTTPException`，但在某些 Starlette 特定的上下文中，可能需要使用 `StarletteHTTPException`。

## 2.5 HTTP 狀態碼與異常的最佳實踐

### 選擇適當的狀態碼

| 操作 | 成功狀態碼 | 常見錯誤狀態碼 |
|------|-----------|---------------|
| 獲取資源 | 200 OK | 404 Not Found, 403 Forbidden |
| 創建資源 | 201 Created | 400 Bad Request, 409 Conflict |
| 更新資源 | 200 OK 或 204 No Content | 400 Bad Request, 404 Not Found |
| 刪除資源 | 204 No Content | 404 Not Found, 403 Forbidden |
| 批量操作 | 200 OK | 400 Bad Request, 422 Unprocessable Entity |

### 提供有用的錯誤信息

良好的錯誤信息應該：

1. 清晰說明錯誤原因
2. 提供足夠的信息幫助調試
3. 避免洩露敏感信息
4. 可能的話，提供解決方案

```python
# 不好的錯誤信息
raise HTTPException(status_code=400, detail="Invalid input")

# 好的錯誤信息
raise HTTPException(
    status_code=400,
    detail="Username must be between 3 and 20 characters and contain only letters and numbers"
)
```

### 結構化錯誤響應

對於複雜的 API，考慮使用結構化的錯誤響應格式：

```python
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.status_code,
            "message": exc.detail,
            "path": request.url.path
        }
    )

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Item ID must be positive"
        )
    if item_id == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found"
        )
    return {"item_id": item_id, "name": f"Item {item_id}"}
```

這將產生如下格式的錯誤響應：

```json
{
  "status": "error",
  "code": 404,
  "message": "Item not found",
  "path": "/items/0"
}
```

## 2.6 狀態碼選擇決策樹

以下決策樹可幫助您選擇適當的 HTTP 狀態碼：

| 問題 | 是 | 否 |
|------|----|----|
| 客戶端請求格式有誤？ | 400 Bad Request | 繼續 |
| 需要身份驗證？ | 401 Unauthorized | 繼續 |
| 已認證但無權訪問？ | 403 Forbidden | 繼續 |
| 請求的資源不存在？ | 404 Not Found | 繼續 |
| 使用了不支持的 HTTP 方法？ | 405 Method Not Allowed | 繼續 |
| 請求與服務器狀態衝突？ | 409 Conflict | 繼續 |
| 請求數據驗證失敗？ | 422 Unprocessable Entity | 繼續 |
| 請求頻率超過限制？ | 429 Too Many Requests | 繼續 |
| 服務器發生錯誤？ | 500 Internal Server Error | 繼續 |
| 服務暫時不可用？ | 503 Service Unavailable | 200 OK |

## 小結

FastAPI 的內建異常類型和 HTTP 狀態碼提供了強大的錯誤處理基礎：

- `HTTPException` 是最常用的異常類型，用於返回特定的 HTTP 狀態碼和錯誤信息
- 使用 `status` 模塊中的常量可提高代碼可讀性
- 選擇適當的 HTTP 狀態碼對於構建符合 RESTful 原則的 API 至關重要
- 提供清晰、結構化的錯誤信息有助於改善 API 的可用性
- 自定義標頭可用於提供額外的錯誤處理指導
