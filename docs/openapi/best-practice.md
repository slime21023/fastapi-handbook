# OpenAPI 與 FastAPI 最佳實踐

## 1. 架構設計原則

### 1.1 API 設計模式

| 模式 | 說明 | 適用場景 |
|-----|------|---------|
| 資源導向 | 以資源為中心設計端點 | CRUD 操作、RESTful API |
| 動作導向 | 以操作為中心設計端點 | 複雜業務流程、RPC 風格 API |
| 混合模式 | 結合資源和動作 | 大型系統、多樣化需求 |

```python
# 資源導向示例
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"id": user_id, "name": "Alice"}

# 動作導向示例
@app.post("/process-payment/")
async def process_payment(payment: PaymentModel):
    return {"status": "processed", "transaction_id": "123"}
```

### 1.2 模組化設計

使用 APIRouter 組織大型 API：

```python
from fastapi import APIRouter, FastAPI

app = FastAPI()

# 用戶相關路由
user_router = APIRouter(prefix="/users", tags=["users"])

@user_router.get("/{user_id}")
async def read_user(user_id: int):
    return {"id": user_id, "name": "Alice"}

# 註冊路由
app.include_router(user_router)
```

### 1.3 版本控制策略

| 策略 | 實現方式 | 優點 | 缺點 |
|-----|---------|-----|-----|
| URL 路徑 | `/v1/users/` | 簡單直觀 | URL 變長 |
| 查詢參數 | `/users/?version=1` | 不改變資源路徑 | 可選性導致複雜性 |
| 標頭 | `X-API-Version: 1` | 保持 URL 清潔 | 不易於瀏覽器直接訪問 |
| 內容協商 | `Accept: application/vnd.api+json;version=1` | 符合 HTTP 標準 | 較複雜 |

```python
# URL 路徑版本控制
@app.get("/v1/users/")
async def read_users_v1():
    return [{"id": 1, "name": "Alice"}]

# v2 API 添加了更多字段
@app.get("/v2/users/")
async def read_users_v2():
    return [{"id": 1, "name": "Alice", "email": "alice@example.com"}]
```

## 2. 數據模型最佳實踐

### 2.1 模型設計原則

| 原則 | 說明 | 示例 |
|-----|------|-----|
| 單一職責 | 每個模型專注於一個領域 | 分離 `UserProfile` 和 `UserCredentials` |
| 繼承與組合 | 利用繼承減少重複代碼 | `BaseItem` → `BookItem`, `ElectronicItem` |
| 驗證邏輯內置 | 在模型中內置驗證邏輯 | 使用 Pydantic 驗證器 |
| 文檔友好 | 添加清晰的字段描述和示例 | 使用 `Field` 的 `description` 和 `example` |

```python
from pydantic import BaseModel, EmailStr, Field, field_validator

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, example="johndoe")
    email: EmailStr = Field(..., example="john@example.com")

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    
    @field_validator("password")
    def password_strength(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError("密碼必須包含至少一個數字")
        return v

class User(UserBase):
    id: int = Field(..., example=1)
    is_active: bool = True
```

### 2.2 請求與響應模型分離

為不同操作設計專用模型：

```python
# 請求模型
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    email: EmailStr = None
    full_name: str = None

# 響應模型
class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    is_active: bool

@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate):
    # 創建用戶邏輯
    return {
        "id": 1,
        "username": user.username,
        "email": user.email,
        "is_active": True
    }
```

### 2.3 通用模式與模型

常用模型模式：

```python
from typing import Generic, List, TypeVar
from pydantic.generics import GenericModel

# 分頁響應
T = TypeVar('T')

class Page(GenericModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int

# 使用示例
@app.get("/users/", response_model=Page[User])
async def read_users(page: int = 1, size: int = 10):
    users = [{"id": i, "username": f"user{i}", "email": f"user{i}@example.com", "is_active": True} 
             for i in range(1, 11)]
    return {
        "items": users,
        "total": 100,
        "page": page,
        "size": size
    }
```

## 3. 路徑操作最佳實踐

### 3.1 HTTP 方法使用指南

| HTTP 方法 | 用途 | 示例 |
|---------|-----|-----|
| GET | 獲取資源 | `GET /users/` 獲取用戶列表 |
| POST | 創建資源 | `POST /users/` 創建新用戶 |
| PUT | 全量更新資源 | `PUT /users/123` 更新整個用戶 |
| PATCH | 部分更新資源 | `PATCH /users/123` 更新部分用戶字段 |
| DELETE | 刪除資源 | `DELETE /users/123` 刪除用戶 |

```python
@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    """創建新項目 (201 Created)"""
    return {"id": 1, **item.model_dump()}

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int):
    """刪除項目 (204 No Content)"""
    if item_id != 1:
        raise HTTPException(status_code=404, detail="Item not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

### 3.2 路徑參數與查詢參數

使用指南：

| 參數類型 | 適用場景 | 示例 |
|---------|---------|-----|
| 路徑參數 | 資源標識符 | `/users/{user_id}` |
| 查詢參數 | 過濾、排序、分頁 | `/users/?role=admin&sort=name` |

```python
@app.get("/users/{user_id}")
async def read_user(
    user_id: int = Path(..., title="用戶 ID", ge=1),
    include_inactive: bool = False
):
    """獲取特定用戶信息"""
    return {"user_id": user_id, "include_inactive": include_inactive}

@app.get("/users/")
async def read_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    role: str = None,
    sort: List[str] = Query([])
):
    """獲取用戶列表"""
    return {
        "skip": skip,
        "limit": limit,
        "role": role,
        "sort": sort
    }
```

### 3.3 狀態碼使用指南

| 狀態碼 | 用途 | FastAPI 實現 |
|-------|-----|-------------|
| 200 OK | 成功獲取資源 | 默認 GET 響應 |
| 201 Created | 成功創建資源 | `status_code=201` |
| 204 No Content | 成功但無返回內容 | `status_code=204` + `Response()` |
| 400 Bad Request | 請求格式錯誤 | 驗證錯誤或 `HTTPException(400)` |
| 401 Unauthorized | 未認證 | `HTTPException(401)` |
| 403 Forbidden | 權限不足 | `HTTPException(403)` |
| 404 Not Found | 資源不存在 | `HTTPException(404)` |

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid item ID"
        )
    elif item_id == 999:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found"
        )
    return {"item_id": item_id, "name": "Example Item"}
```

## 4. 安全與認證最佳實踐

### 4.1 認證方案對比

| 認證方案 | 適用場景 | 實現方式 |
|---------|---------|---------|
| API 密鑰 | 簡單的 API 訪問控制 | 請求標頭、查詢參數或 Cookie |
| OAuth2 密碼流 | 用戶名/密碼登錄 | `OAuth2PasswordBearer` |
| JWT 令牌 | 無狀態身份驗證 | `OAuth2PasswordBearer` + JWT 編碼/解碼 |

```python
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader

# OAuth2 密碼流
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != "johndoe" or form_data.password != "secret":
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": "fake_token", "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    return {"username": "johndoe", "token": token}

# API 密鑰
api_key_header = APIKeyHeader(name="X-API-Key")

@app.get("/items/")
async def read_items(api_key: str = Depends(api_key_header)):
    if api_key != "valid_api_key":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return [{"id": 1, "name": "Item 1"}]
```

### 4.2 JWT 認證實現

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt

# JWT 配置
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"username": username}

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # 驗證用戶
    if form_data.username != "johndoe" or form_data.password != "secret":
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    # 創建訪問令牌
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}
```

### 4.3 權限控制

```python
from fastapi import Security
from fastapi.security import SecurityScopes

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "users:read": "讀取用戶信息",
        "users:write": "修改用戶信息",
        "items:read": "讀取項目",
        "items:write": "創建或修改項目",
    },
)

async def get_current_user_with_scopes(
    security_scopes: SecurityScopes, 
    token: str = Depends(oauth2_scheme)
):
    # 簡化示例，實際應解析 JWT 令牌
    if token != "valid_token":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # 假設令牌包含這些範圍
    token_scopes = ["users:read", "items:read"]
    
    # 檢查所需範圍
    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Not enough permissions. Required: {scope}",
            )
    
    return {"username": "johndoe", "scopes": token_scopes}

@app.get(
    "/users/", 
    dependencies=[Security(get_current_user_with_scopes, scopes=["users:read"])]
)
async def read_users():
    return [{"username": "johndoe"}]

@app.post(
    "/users/", 
    dependencies=[Security(get_current_user_with_scopes, scopes=["users:write"])]
)
async def create_user(username: str):
    return {"username": username}
```

## 5. 錯誤處理最佳實踐

### 5.1 標準化錯誤響應

```python
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict = None

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code="VALIDATION_ERROR",
            message="資料驗證錯誤",
            details={"errors": exc.errors()}
        ).model_dump(),
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            details=exc.headers if hasattr(exc, "headers") else None
        ).model_dump(),
    )

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 0:
        raise HTTPException(status_code=400, detail="Invalid item ID")
    if item_id == 999:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "name": "Sample Item"}
```

### 5.2 自定義異常

```python
class NotFoundError(Exception):
    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.message = f"{resource_type} with ID {resource_id} not found"
        super().__init__(self.message)

@app.exception_handler(NotFoundError)
async def not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            code="RESOURCE_NOT_FOUND",
            message=exc.message,
            details={
                "resource_type": exc.resource_type,
                "resource_id": exc.resource_id
            }
        ).model_dump(),
    )

@app.get("/users/{user_id}")
async def read_user(user_id: int):
    if user_id == 999:
        raise NotFoundError("User", str(user_id))
    return {"id": user_id, "name": "Sample User"}
```

## 6. 總結

### 6.1 OpenAPI 與 FastAPI 最佳實踐清單

| 類別 | 最佳實踐 |
|-----|---------|
| API 設計 | 使用資源導向設計、模組化路由、一致的版本控制 |
| 數據模型 | 分離請求/響應模型、使用繼承減少重複、添加完整驗證 |
| 路徑操作 | 正確使用 HTTP 方法、適當使用狀態碼、清晰的參數設計 |
| 安全性 | 實現 OAuth2/JWT 認證、精細的權限控制、安全的密碼處理 |
| 錯誤處理 | 標準化錯誤響應、自定義異常處理、完整的錯誤信息 |

### 6.2 設計原則

1. **一致性** - 保持 API 設計風格一致
2. **簡潔性** - 保持 API 簡單明了
3. **可擴展性** - 設計易於擴展的 API
4. **安全性** - 從設計階段考慮安全性
5. **文檔優先** - 優先考慮 API 文檔的清晰度

透過遵循這些最佳實踐，您可以使用 FastAPI 和 OpenAPI 規範構建高效、可靠且易於維護的 API。這些實踐將幫助您提高開發效率，改善用戶體驗，並確保您的 API 具有良好的可擴展性和可維護性。
