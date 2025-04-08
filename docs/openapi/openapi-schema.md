# OpenAPI Schema 與 FastAPI/Pydantic 的整合

## 第一部分：基本概念

### OpenAPI Schema 簡介

OpenAPI Schema (以前稱為 Swagger Specification) 是一種用於描述 REST API 的標準化格式。它提供了一種機器可讀的方式來描述 API 的端點、參數、請求體、響應、認證方式等。

| 核心概念 | 說明 |
|---------|------|
| **路徑** | API 端點的 URL 路徑和 HTTP 方法 |
| **參數** | 路徑、查詢、頭部和 cookie 參數 |
| **請求體** | HTTP 請求中的數據結構 |
| **響應** | 不同狀態碼的響應數據結構 |
| **組件** | 可重用的 schema 定義 |

### FastAPI 與 OpenAPI 的整合

FastAPI 自動生成 OpenAPI Schema，並提供兩種交互式文檔界面：Swagger UI 和 ReDoc。

```python
from fastapi import FastAPI

app = FastAPI(
    title="我的 API",
    description="這是一個示例 API",
    version="0.1.0",
    openapi_url="/api/openapi.json",  # OpenAPI Schema 的路徑
    docs_url="/docs",                 # Swagger UI 的路徑
    redoc_url="/redoc"                # ReDoc 的路徑
)
```

### 訪問自動生成的文檔

| 文檔界面 | 默認 URL | 特點 |
|---------|---------|------|
| Swagger UI | `/docs` | 互動式測試界面 |
| ReDoc | `/redoc` | 易讀的文檔格式 |
| OpenAPI JSON | `/openapi.json` | 原始 OpenAPI Schema |

### Pydantic 模型與 JSON Schema

Pydantic 模型自動轉換為 JSON Schema，這是 OpenAPI Schema 的一部分。

#### 基本類型映射

| Python 類型 | JSON Schema 類型 |
|------------|-----------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |
| `dict` | `object` |
| `None` | `null` |

#### 模型示例

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Item(BaseModel):
    id: int = Field(description="項目的唯一標識符")
    name: str = Field(min_length=1, max_length=100, description="項目名稱")
    price: float = Field(gt=0, description="項目價格")
    tags: List[str] = Field(default=[], description="項目標籤")
    description: Optional[str] = Field(None, description="項目詳細描述")
```

### 路徑操作裝飾器與 OpenAPI

FastAPI 的路徑操作裝飾器自動生成 OpenAPI 路徑項。

```python
from fastapi import FastAPI, Path, Query, Body
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None

@app.get(
    "/items/{item_id}",
    summary="獲取單個項目",
    description="根據 ID 獲取項目的詳細信息",
    response_description="項目詳細信息",
    tags=["items"],
    response_model=Item
)
async def read_item(
    item_id: int = Path(..., title="項目 ID", ge=1),
    q: Optional[str] = Query(None, min_length=3, max_length=50, description="搜索查詢")
):
    """
    獲取單個項目的詳細信息:
    
    - **item_id**: 項目的唯一標識符
    - **q**: 可選的搜索查詢參數
    """
    return {"name": "範例項目", "price": 45.5, "description": "這是一個範例項目"}
```

### 路徑參數增強

使用 `Path` 類增強路徑參數的 OpenAPI 文檔：

```python
from fastapi import FastAPI, Path

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(
    item_id: int = Path(
        ...,
        title="項目 ID",
        description="要獲取的項目的 ID",
        ge=1,
        le=1000,
        example=42
    )
):
    return {"item_id": item_id}
```

### 查詢參數增強

使用 `Query` 類增強查詢參數的 OpenAPI 文檔：

```python
from fastapi import FastAPI, Query
from typing import List, Optional

app = FastAPI()

@app.get("/items/")
async def read_items(
    q: Optional[str] = Query(
        None,
        title="查詢字符串",
        description="用於過濾項目的查詢字符串",
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_-]*$",
        deprecated=False,
        example="example_query"
    ),
    skip: int = Query(0, ge=0, description="要跳過的項目數"),
    limit: int = Query(10, ge=1, le=100, description="要返回的項目數")
):
    return {"q": q, "skip": skip, "limit": limit}
```

### 請求體增強

使用 `Body` 類增強請求體參數的 OpenAPI 文檔：

```python
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

class Item(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, example="Foo")
    description: Optional[str] = Field(None, max_length=300, example="A very nice Item")
    price: float = Field(..., gt=0, example=35.4)

@app.post("/items/")
async def create_item(
    item: Item = Body(
        ...,
        embed=True,
        example={
            "name": "Foo",
            "description": "A very nice Item",
            "price": 35.4
        }
    )
):
    return item
```

### 響應模型與狀態碼

定義不同狀態碼的響應模型：

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Union

app = FastAPI()

class Item(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str] = None

class Message(BaseModel):
    message: str

@app.get(
    "/items/{item_id}",
    response_model=Item,
    responses={
        200: {
            "description": "成功獲取項目",
            "model": Item
        },
        404: {
            "description": "項目未找到",
            "model": Message
        },
        500: {
            "description": "服務器錯誤",
            "model": Message
        }
    }
)
async def read_item(item_id: int):
    if item_id == 404:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="項目未找到"
        )
    elif item_id == 500:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服務器錯誤"
        )
    return {
        "id": item_id,
        "name": "範例項目",
        "price": 45.5,
        "description": "這是一個範例項目"
    }
```

### OpenAPI Schema 與 FastAPI 的關係

FastAPI 使用 Pydantic 模型和路徑操作裝飾器自動生成 OpenAPI Schema。這種關係可以總結為：

1. **Pydantic 模型** → **JSON Schema**：Pydantic 模型被轉換為 JSON Schema，定義了數據結構。
2. **路徑操作裝飾器** → **OpenAPI 路徑項**：路徑操作裝飾器定義了 API 端點，包括 URL、HTTP 方法和參數。
3. **參數裝飾器** → **OpenAPI 參數**：`Path`、`Query`、`Body` 等裝飾器增強了參數的描述和驗證。
4. **響應模型** → **OpenAPI 響應**：`response_model` 定義了 API 端點的響應結構。
5. **依賴項** → **OpenAPI 安全性和參數**：依賴項可以定義安全要求和額外參數。

### 基本總結

FastAPI 與 Pydantic 緊密結合，提供了強大的 API 文檔生成功能：

1. **自動文檔生成**：根據代碼自動生成 API 文檔
2. **類型安全**：通過 Pydantic 模型確保數據類型安全
3. **參數驗證**：自動驗證請求參數和請求體
4. **響應模型**：明確定義 API 響應的數據結構
5. **交互式文檔**：提供 Swagger UI 和 ReDoc 兩種交互式文檔界面

通過這些基本功能，開發者可以輕鬆創建具有完整文檔的 API，同時確保數據驗證和類型安全。