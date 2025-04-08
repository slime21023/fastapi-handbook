# FastAPI 基礎使用

## 安裝與環境設置

### 安裝 FastAPI

```bash
pip install fastapi[all]
```

| 安裝選項 | 說明 | 命令 |
|---------|------|------|
| 最小安裝 | 僅安裝 FastAPI 核心功能 | `pip install fastapi` |
| 標準安裝 | 安裝 FastAPI 和 ASGI 伺服器 | `pip install fastapi uvicorn` |
| 完整安裝 | 安裝所有依賴和可選功能 | `pip install fastapi[all]` |

### 運行應用

```bash
uvicorn app.main:app --reload
```

| 參數 | 說明 |
|------|------|
| `app.main:app` | 模組路徑:應用實例（main.py 中的 app 變數） |
| `--reload` | 開發模式，代碼變更時自動重新載入 |

## 創建第一個 API

```python
from fastapi import FastAPI

app = FastAPI(
    title="我的第一個 API",
    description="這是使用 FastAPI 創建的簡單 API",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

### 訪問自動生成的文檔

| 文檔界面 | URL | 特點 |
|---------|-----|------|
| Swagger UI | http://127.0.0.1:8000/docs | 互動式測試界面 |
| ReDoc | http://127.0.0.1:8000/redoc | 易讀的文檔格式 |
| OpenAPI JSON | http://127.0.0.1:8000/openapi.json | 原始 OpenAPI Schema |

## 路徑參數 (Path Parameters)

路徑參數是 URL 的一部分，用於標識特定資源：

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

| 特點 | 說明 |
|------|------|
| **類型轉換** | FastAPI 會自動將參數轉換為指定類型 |
| **驗證** | 如果轉換失敗，會自動返回適當的錯誤響應 |
| **文檔** | 參數類型會反映在 API 文檔中 |

### 路徑參數增強

使用 Path 類增加路徑參數的驗證和元數據：

```python
from fastapi import Path

@app.get("/items/{item_id}")
async def read_items(
    item_id: int = Path(
        ...,
        title="項目 ID",
        description="要獲取的項目的 ID",
        gt=0,
        le=1000
    )
):
    return {"item_id": item_id}
```

## 查詢參數 (Query Parameters)

查詢參數是 URL 中 `?` 後面的部分，用於過濾、排序或分頁：

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10, q: str = None):
    return {"skip": skip, "limit": limit, "q": q}
```

| 特點 | 說明 |
|------|------|
| **可選參數** | 設置默認值使參數變為可選 |
| **類型註解** | 使用 Python 類型註解定義參數類型 |
| **自動驗證** | FastAPI 自動驗證參數類型和約束 |

### 查詢參數驗證

使用 Query 類增加更多驗證和元數據：

```python
from fastapi import Query

@app.get("/items/")
async def read_items(
    q: str = Query(
        None,
        min_length=3,
        max_length=50,
        regex="^[a-z]+$",
        title="查詢字符串",
        description="用於過濾項目的查詢字符串"
    )
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
```

| 驗證選項 | 說明 |
|---------|------|
| `min_length` | 最小長度 |
| `max_length` | 最大長度 |
| `regex` | 正則表達式模式 |
| `gt` / `ge` | 大於 / 大於等於（數值） |
| `lt` / `le` | 小於 / 小於等於（數值） |

## 請求體 (Request Body)

使用 Pydantic 模型定義請求體：

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

| 特點 | 說明 |
|------|------|
| **自動解析** | JSON 請求體自動解析為 Python 對象 |
| **類型驗證** | 自動驗證數據類型和約束 |
| **文檔生成** | 模型結構自動反映在 API 文檔中 |

## HTTP 方法

FastAPI 提供了對應各種 HTTP 方法的裝飾器：

| HTTP 方法 | FastAPI 裝飾器 | 常見用途 |
|----------|---------------|---------|
| GET | `@app.get()` | 獲取資源 |
| POST | `@app.post()` | 創建資源 |
| PUT | `@app.put()` | 更新資源（完整替換） |
| PATCH | `@app.patch()` | 部分更新資源 |
| DELETE | `@app.delete()` | 刪除資源 |

## 狀態碼

```python
from fastapi import status

@app.post(
    "/items/",
    status_code=status.HTTP_201_CREATED
)
async def create_item(item: Item):
    return item
```

## 錯誤處理

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found"
        )
    return {"item": items[item_id]}
```

## 路由器與 API 組織

使用 APIRouter 組織大型應用：

```python
from fastapi import APIRouter

router = APIRouter(
    prefix="/items",
    tags=["items"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def read_items():
    return [{"name": "Item 1"}, {"name": "Item 2"}]

@router.get("/{item_id}")
async def read_item(item_id: str):
    return {"name": "Item", "item_id": item_id}

# 在主應用中包含路由器
app = FastAPI()
app.include_router(router)
```

