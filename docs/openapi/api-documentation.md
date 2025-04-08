# API 文檔最佳實踐：FastAPI 整合

## 1. FastAPI 文檔系統概述

### 1.1 內建文檔功能

FastAPI 提供自動生成的互動式 API 文檔：

| 文檔介面 | 路徑 | 特點 |
|---------|-----|------|
| Swagger UI | `/docs` | 互動式測試、豐富視覺效果 |
| ReDoc | `/redoc` | 更清晰的閱讀體驗、更好的導航 |

```python
from fastapi import FastAPI

app = FastAPI(
    title="我的 API",
    description="這是一個示範 API 文檔的示例",
    version="0.1.0",
    docs_url="/documentation",  # 自訂 Swagger UI 路徑
    redoc_url="/redocumentation"  # 自訂 ReDoc 路徑
)

@app.get("/")
def read_root():
    return {"Hello": "World"}
```

### 1.2 OpenAPI 規範整合

FastAPI 自動生成符合 OpenAPI 規範的 JSON Schema：

```python
# 訪問 /openapi.json 獲取完整的 OpenAPI 規範
# 可用於其他工具和文檔生成器
```

## 2. 基本文檔元素

### 2.1 API 元數據

設定 API 的基本信息：

```python
from fastapi import FastAPI

app = FastAPI(
    title="產品管理系統",
    description="""
    # 產品管理系統 API
    
    這個 API 允許您：
    * 創建產品
    * 查詢產品
    * 更新產品
    * 刪除產品
    
    ## 注意事項
    所有操作需要適當的權限。
    """,
    version="1.0.0",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "API 支援團隊",
        "url": "http://example.com/contact/",
        "email": "support@example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }
)
```

### 2.2 路徑操作描述

為每個端點添加詳細描述：

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post(
    "/items/",
    summary="創建新項目",
    description="""
    創建一個新項目，包含以下信息：
    - **name**: 項目名稱
    - **price**: 項目價格
    
    價格必須大於零。
    """,
    response_description="創建成功的項目信息"
)
async def create_item(item: Item):
    """
    創建項目：
    
    - **name**: 每個項目必須有一個名稱
    - **price**: 必須是正數
    """
    return item
```

### 2.3 標籤與分組

使用標籤組織 API 端點：

```python
from fastapi import FastAPI

app = FastAPI(
    openapi_tags=[
        {
            "name": "users",
            "description": "用戶管理操作",
            "externalDocs": {
                "description": "用戶相關外部文檔",
                "url": "https://example.com/docs/users/",
            },
        },
        {
            "name": "items",
            "description": "項目管理操作",
        },
    ]
)

@app.get("/users/", tags=["users"])
async def read_users():
    return [{"name": "Harry"}]

@app.get("/items/", tags=["items"])
async def read_items():
    return [{"name": "Wand"}]

@app.get("/both/", tags=["users", "items"])
async def read_both():
    return {"users": [{"name": "Harry"}], "items": [{"name": "Wand"}]}
```

## 3. 高級文檔技巧

### 3.1 請求體示例

提供請求體的示例：

```python
from fastapi import Body, FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Foo",
                    "price": 35.4
                }
            ]
        }
    }

@app.post("/items/")
async def create_item(
    item: Item = Body(
        ...,
        examples=[
            {
                "summary": "標準項目",
                "description": "一個標準價格的項目示例",
                "value": {
                    "name": "Foo",
                    "price": 35.4
                }
            },
            {
                "summary": "高級項目",
                "description": "一個高價格的項目示例",
                "value": {
                    "name": "Bar",
                    "price": 62.0
                }
            }
        ]
    )
):
    return item
```

### 3.2 響應示例

定義不同狀態碼的響應模型和示例：

```python
from typing import Dict, Union
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    id: str
    name: str
    price: float

class Message(BaseModel):
    message: str

@app.get(
    "/items/{item_id}",
    response_model=Item,
    responses={
        200: {
            "description": "成功獲取項目",
            "content": {
                "application/json": {
                    "example": {"id": "foo", "name": "Foo", "price": 50.2}
                }
            }
        },
        404: {
            "description": "項目未找到",
            "model": Message,
            "content": {
                "application/json": {
                    "example": {"message": "找不到此項目"}
                }
            }
        }
    }
)
async def read_item(item_id: str):
    if item_id != "foo":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="找不到此項目"
        )
    return {"id": "foo", "name": "Foo", "price": 50.2}
```

### 3.3 棄用標記

標記已棄用的端點：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/legacy/", deprecated=True)
async def read_legacy():
    return {"message": "這個端點已棄用"}

@app.get("/current/")
async def read_current():
    return {"message": "這是當前版本的端點"}
```

## 4. 文檔與代碼的結合

### 4.1 文檔字符串 (Docstrings)

利用 Python 文檔字符串增強文檔：

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    """
    創建新項目
    
    此端點允許創建新的項目記錄。
    
    - **name**: 項目名稱，不能為空
    - **price**: 項目價格，必須大於零
    
    返回創建的項目，包括生成的 ID。
    """
    return {"id": "123", **item.model_dump()}
```

### 4.2 類型註解與文檔

利用類型註解提高文檔質量：

```python
from typing import Dict, List, Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.get("/items/", response_model=List[Item])
async def read_items() -> List[Item]:
    """
    獲取所有項目
    
    返回系統中所有項目的列表。
    """
    return [
        {"name": "Item1", "price": 50.2},
        {"name": "Item2", "price": 30, "description": "This is Item2"}
    ]

@app.get("/items/{item_id}", response_model=Union[Item, Dict[str, str]])
async def read_item(item_id: str) -> Union[Item, Dict[str, str]]:
    """
    獲取特定項目
    
    如果找到項目，返回項目詳情；否則返回錯誤消息。
    """
    if item_id == "foo":
        return {"name": "Foo", "price": 50.2}
    return {"message": "Item not found"}
```

## 5. 文檔自定義與擴展

### 5.1 自定義 OpenAPI 操作 ID

指定操作 ID 以便客戶端代碼生成：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/", operation_id="get_items_list")
async def read_items():
    return [{"name": "Foo"}]

@app.get("/items/{item_id}", operation_id="get_item_by_id")
async def read_item(item_id: str):
    return {"name": "Foo", "item_id": item_id}
```

### 5.2 擴展 OpenAPI Schema

添加自定義 OpenAPI 擴展：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get(
    "/items/",
    openapi_extra={
        "x-custom-extension": "value",
        "x-rate-limit": {
            "max": 100,
            "period": "hour"
        }
    }
)
async def read_items():
    return [{"name": "Foo"}]
```

### 5.3 完全自定義 OpenAPI Schema

完全控制 OpenAPI Schema：

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()

@app.get("/items/")
async def read_items():
    return [{"name": "Foo"}]

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="自定義標題",
        version="2.5.0",
        description="這是一個自定義的 OpenAPI schema",
        routes=app.routes,
    )
    
    # 自定義路徑
    openapi_schema["paths"]["/items/"]["get"]["summary"] = "讀取項目列表"
    
    # 添加自定義標籤
    openapi_schema["tags"] = [
        {
            "name": "items",
            "description": "項目操作",
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

## 6. 文檔互動功能

### 6.1 授權配置

配置 Swagger UI 的授權功能：

```python
from fastapi import Depends, FastAPI, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # 簡化的驗證邏輯
    if form_data.username == "user" and form_data.password == "password":
        return {"access_token": "fake_token", "token_type": "bearer"}
    return {"error": "Invalid credentials"}

@app.get("/items/")
async def read_items(token: str = Depends(oauth2_scheme)):
    return [{"name": "Foo"}]
```

### 6.2 自定義 Swagger UI 和 ReDoc

自定義文檔界面：

```python
from fastapi import FastAPI

app = FastAPI(
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,  # 隱藏 Models 部分
        "displayRequestDuration": True,  # 顯示請求持續時間
        "docExpansion": "none",  # 默認摺疊所有操作
        "filter": True,  # 啟用過濾功能
        "syntaxHighlight.theme": "agate",  # 語法高亮主題
    }
)

@app.get("/items/")
async def read_items():
    return [{"name": "Foo"}]
```

## 7. 文檔版本控制

### 7.1 API 版本控制

使用路徑參數進行版本控制：

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# v1 API
v1_router = APIRouter(prefix="/v1", tags=["v1"])

@v1_router.get("/items/")
async def read_items_v1():
    return [{"name": "Foo", "version": "v1"}]

# v2 API
v2_router = APIRouter(prefix="/v2", tags=["v2"])

@v2_router.get("/items/")
async def read_items_v2():
    return [{"name": "Foo", "version": "v2", "extra": "data"}]

app.include_router(v1_router)
app.include_router(v2_router)
```

### 7.2 棄用與過渡

標記棄用的 API 版本：

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# 已棄用的 v1 API
v1_router = APIRouter(prefix="/v1", tags=["v1-deprecated"])

@v1_router.get("/items/", deprecated=True)
async def read_items_v1():
    """
    此端點已棄用，將在 2023 年 12 月 31 日移除。
    請使用 /v2/items/ 端點。
    """
    return [{"name": "Foo", "version": "v1"}]

# 當前 v2 API
v2_router = APIRouter(prefix="/v2", tags=["v2-current"])

@v2_router.get("/items/")
async def read_items_v2():
    """
    獲取項目列表（當前版本）
    """
    return [{"name": "Foo", "version": "v2", "extra": "data"}]

app.include_router(v1_router)
app.include_router(v2_router)
```

## 8. 文檔最佳實踐

### 8.1 文檔清晰度檢查表

| 項目 | 建議 | 範例 |
|-----|------|-----|
| 端點摘要 | 簡短、動詞開頭 | "獲取用戶列表" |
| 端點描述 | 詳細說明功能和限制 | "返回系統中的所有活躍用戶，分頁顯示" |
| 參數描述 | 說明用途、格式和約束 | "用戶 ID，必須是有效的 UUID 格式" |
| 響應描述 | 說明返回數據結構和狀態碼 | "返回用戶詳細信息，包括個人資料和權限" |
| 示例值 | 提供有意義的示例 | `{"name": "張三", "email": "zhang@example.com"}` |
| 錯誤處理 | 描述可能的錯誤和解決方法 | "404: 用戶不存在，請檢查 ID 是否正確" |

### 8.2 文檔組織建議

```python
from fastapi import FastAPI, APIRouter

app = FastAPI(
    title="產品管理系統",
    description="""
    # 產品管理系統 API 文檔
    
    ## 功能模塊
    * 用戶管理 - 處理用戶帳戶和認證
    * 產品管理 - 處理產品的 CRUD 操作
    * 訂單管理 - 處理訂單流程
    
    ## 使用指南
    所有請求需要在標頭中包含有效的 API 密鑰。
    """
)

# 用戶相關路由
user_router = APIRouter(
    prefix="/users",
    tags=["用戶管理"],
    responses={404: {"description": "用戶未找到"}}
)

# 產品相關路由
product_router = APIRouter(
    prefix="/products",
    tags=["產品管理"],
    responses={404: {"description": "產品未找到"}}
)

# 訂單相關路由
order_router = APIRouter(
    prefix="/orders",
    tags=["訂單管理"],
    responses={404: {"description": "訂單未找到"}}
)

# 添加路由
app.include_router(user_router)
app.include_router(product_router)
app.include_router(order_router)
```

### 8.3 文檔維護策略

| 策略 | 說明 | 實施方式 |
|-----|------|---------|
| 文檔審查 | 定期審查文檔的準確性和完整性 | 建立文檔審查清單和流程 |
| 自動化測試 | 確保文檔示例與實際 API 行為一致 | 使用文檔示例作為測試用例 |
| 變更記錄 | 記錄 API 變更和文檔更新 | 維護 CHANGELOG.md 文件 |
| 文檔版本 | 將文檔版本與 API 版本同步 | 在文檔中標明適用的 API 版本 |

## 9. 文檔工具整合

### 9.1 API 客戶端生成

從 OpenAPI 規範生成客戶端代碼：

| 工具 | 支持語言 | 用法 |
|-----|---------|-----|
| OpenAPI Generator | 多種語言 | `openapi-generator generate -i openapi.json -g python -o ./client` |
| Swagger Codegen | 多種語言 | `swagger-codegen generate -i openapi.json -l typescript-fetch -o ./client` |
| TypeScript | TypeScript | 使用 `openapi-typescript` 包 |

### 9.2 文檔導出

將 API 文檔導出為其他格式：

```python
# 導出 OpenAPI 規範為 JSON 文件
import json
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
async def read_items():
    return [{"name": "Foo"}]

with open("openapi.json", "w") as f:
    json.dump(app.openapi(), f, indent=2)
```

### 9.3 第三方文檔工具

與其他文檔工具整合：

| 工具 | 用途 | 整合方式 |
|-----|------|---------|
| Postman | API 測試與文檔 | 導入 OpenAPI 規範 |
| Stoplight | API 設計與文檔 | 導入/導出 OpenAPI 規範 |
| ReadMe | API 文檔門戶 | 同步 OpenAPI 規範 |

## 10. 總結與最佳實踐

### 10.1 文檔優化技巧

| 技巧 | 說明 |
|-----|------|
| 保持一致性 | 使用一致的術語、格式和結構 |
| 面向使用者 | 從 API 消費者的角度撰寫文檔 |
| 提供示例 | 為每個端點提供實用的請求和響應示例 |
| 說明業務邏輯 | 解釋 API 行為背後的業務邏輯 |
| 文檔即代碼 | 將文檔視為代碼的一部分，隨代碼一起維護 |

### 10.2 常見問題與解決方案

| 問題 | 解決方案 |
|-----|---------|
| 文檔與代碼不同步 | 將文檔嵌入代碼中，使用自動生成工具 |
| 文檔過於技術化 | 增加業務說明，使用非技術語言 |
| 缺乏使用示例 | 為每個端點添加實用的示例 |
| 安全信息暴露 | 使用環境變量，避免在文檔中包含敏感信息 |
| 文檔難以導航 | 使用標籤和分組組織端點 |

### 10.3 文檔驅動開發

1. 先設計 API 文檔
2. 根據文檔實現 API
3. 使用文檔驗證實現
4. 迭代改進文檔和實現

```python
# 文檔驅動開發示例
from fastapi import FastAPI
from pydantic import BaseModel

# 1. 先定義數據模型和端點規範
class Item(BaseModel):
    name: str
    price: float
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "Foo", "price": 35.4}
            ]
        }
    }

# 2. 根據規範實現 API
app = FastAPI(
    title="項目 API",
    description="基於文檔驅動開發的項目 API"
)

@app.post(
    "/items/",
    summary="創建新項目",
    description="創建一個新的項目記錄",
    response_description="返回創建的項目"
)
async def create_item(item: Item):
    """
    創建新項目:
    
    - 項目名稱不能為空
    - 價格必須大於零
    """
    return item
```

透過這些最佳實踐，您可以創建清晰、完整且易於使用的 API 文檔，提高開發效率並改善 API 使用體驗。