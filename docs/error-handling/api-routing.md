# 5. API Router 與異常處理

在大型 FastAPI 應用中，通常會使用 APIRouter 來組織和模塊化路由。APIRouter 允許將相關的端點分組到不同的路由器中，使代碼更加結構化和可維護。本章將探討如何在使用 APIRouter 時實現有效的異常處理策略。

## 5.1 APIRouter 基礎知識

APIRouter 是 FastAPI 提供的一個工具，用於將相關的端點組織在一起，並可以在主應用中註冊這些路由器。

### 基本用法

```python
from fastapi import APIRouter, FastAPI

# 創建路由器
router = APIRouter(
    prefix="/items",
    tags=["items"],
    responses={404: {"description": "Item not found"}}
)

# 在路由器上定義端點
@router.get("/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id, "name": f"Item {item_id}"}

# 創建主應用
app = FastAPI()

# 將路由器包含在應用中
app.include_router(router)
```

在這個例子中，我們創建了一個前綴為 `/items` 的路由器，並定義了一個端點來讀取項目。然後，我們將該路由器包含在主應用中。

## 5.2 路由器級別的異常處理

每個 APIRouter 可以有自己的異常處理器，用於處理該路由器中的端點拋出的異常。

### 為路由器定義異常處理器

```python
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 創建路由器
router = APIRouter(
    prefix="/users",
    tags=["users"]
)

# 自定義異常
class UserNotFoundError(Exception):
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.message = f"User with ID {user_id} not found"
        super().__init__(self.message)

# 路由器級別的異常處理器
@router.exception_handler(UserNotFoundError)
async def user_not_found_exception_handler(request: Request, exc: UserNotFoundError):
    logger.error(f"User not found: {exc.user_id}")
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": exc.message,
            "user_id": exc.user_id
        }
    )

# 路由器端點
@router.get("/{user_id}")
async def read_user(user_id: int):
    if user_id <= 0:
        raise UserNotFoundError(user_id)
    return {"user_id": user_id, "name": f"User {user_id}"}

# 創建主應用
app = FastAPI()

# 將路由器包含在應用中
app.include_router(router)
```

在這個例子中，我們為 `/users` 路由器定義了一個異常處理器，專門處理 `UserNotFoundError` 異常。當路由器中的端點拋出這個異常時，將由這個處理器處理。

## 5.3 路由器與全局異常處理的關係

當一個應用包含多個路由器時，異常處理遵循以下優先順序：

1. 路由器級別的異常處理器
2. 應用級別的異常處理器
3. FastAPI 默認的異常處理

這意味著如果路由器定義了一個異常處理器，那麼該處理器將優先於應用級別的處理器。如果路由器沒有定義處理器，則異常將傳遞給應用級別的處理器。

```python
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 創建路由器
users_router = APIRouter(
    prefix="/users",
    tags=["users"]
)

# 自定義異常
class UserNotFoundError(Exception):
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.message = f"User with ID {user_id} not found"
        super().__init__(self.message)

# 路由器級別的異常處理器
@users_router.exception_handler(UserNotFoundError)
async def user_not_found_exception_handler(request: Request, exc: UserNotFoundError):
    logger.error(f"User not found: {exc.user_id}")
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": exc.message,
            "user_id": exc.user_id
        }
    )

# 路由器端點
@users_router.get("/{user_id}")
async def read_user(user_id: int):
    if user_id <= 0:
        raise UserNotFoundError(user_id)
    return {"user_id": user_id, "name": f"User {user_id}"}

# 創建另一個路由器
items_router = APIRouter(
    prefix="/items",
    tags=["items"]
)

# 自定義異常
class ItemNotFoundError(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id
        self.message = f"Item with ID {item_id} not found"
        super().__init__(self.message)

# 路由器端點
@items_router.get("/{item_id}")
async def read_item(item_id: int):
    if item_id <= 0:
        raise ItemNotFoundError(item_id)
    return {"item_id": item_id, "name": f"Item {item_id}"}

# 創建主應用
app = FastAPI()

# 應用級別的異常處理器
@app.exception_handler(ItemNotFoundError)
async def item_not_found_exception_handler(request: Request, exc: ItemNotFoundError):
    logger.error(f"Item not found: {exc.item_id}")
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": exc.message,
            "item_id": exc.item_id
        }
    )

# 將路由器包含在應用中
app.include_router(users_router)
app.include_router(items_router)
```

在這個例子中：
- `UserNotFoundError` 由 `users_router` 的異常處理器處理
- `ItemNotFoundError` 由應用級別的異常處理器處理，因為 `items_router` 沒有為該異常定義處理器

## 5.4 在路由器中使用依賴項進行異常處理

依賴項是 FastAPI 的一個強大功能，可以用於實現橫切關注點，如身份驗證、授權和輸入驗證。您可以在路由器級別定義依賴項，這些依賴項將應用於該路由器中的所有端點。

### 使用路由器依賴項進行錯誤處理

```python
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Header
from typing import Optional
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 創建路由器
router = APIRouter(
    prefix="/admin",
    tags=["admin"]
)

# 授權依賴項
def verify_admin_token(x_token: Optional[str] = Header(None)):
    if not x_token:
        raise HTTPException(status_code=401, detail="X-Token header missing")
    if x_token != "admin-secret-token":
        raise HTTPException(status_code=403, detail="Invalid admin token")
    return x_token

# 將依賴項應用於整個路由器
router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(verify_admin_token)],
    responses={
        401: {"description": "Missing token"},
        403: {"description": "Invalid token"}
    }
)

# 路由器端點
@router.get("/dashboard")
async def admin_dashboard():
    return {"status": "success", "message": "Welcome to admin dashboard"}

@router.get("/users")
async def list_all_users():
    return {"status": "success", "users": ["user1", "user2", "user3"]}

# 創建主應用
app = FastAPI()

# 將路由器包含在應用中
app.include_router(router)
```

在這個例子中，我們為 `/admin` 路由器定義了一個依賴項 `verify_admin_token`，用於驗證管理員令牌。該依賴項將應用於路由器中的所有端點，如果驗證失敗，將拋出 `HTTPException`。

## 5.5 路由器特定的錯誤響應

您可以在路由器級別定義默認的錯誤響應，這些響應將包含在 API 文檔中：

```python
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

# 錯誤響應模型
class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

# 創建路由器
router = APIRouter(
    prefix="/products",
    tags=["products"],
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Product not found",
            "content": {
                "application/json": {
                    "example": {"status": "error", "message": "Product not found"}
                }
            }
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad request",
            "content": {
                "application/json": {
                    "example": {"status": "error", "message": "Invalid product data"}
                }
            }
        }
    }
)

# 路由器端點
@router.get("/{product_id}")
async def read_product(product_id: int):
    if product_id <= 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"product_id": product_id, "name": f"Product {product_id}"}

# 創建主應用
app = FastAPI()

# 將路由器包含在應用中
app.include_router(router)
```

在這個例子中，我們為 `/products` 路由器定義了兩個默認的錯誤響應：404 和 400。這些響應將顯示在 API 文檔中，幫助 API 消費者理解可能的錯誤情況。

## 5.6 路由器與錯誤碼的標準化

在大型應用中，保持錯誤碼的一致性非常重要。您可以創建一個錯誤碼模塊，並在所有路由器中使用它：

```python
from enum import Enum
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 錯誤碼枚舉
class ErrorCode(str, Enum):
    # 用戶錯誤
    USER_NOT_FOUND = "USER_NOT_FOUND"
    USER_ALREADY_EXISTS = "USER_ALREADY_EXISTS"
    INVALID_USER_DATA = "INVALID_USER_DATA"
    
    # 項目錯誤
    ITEM_NOT_FOUND = "ITEM_NOT_FOUND"
    ITEM_ALREADY_EXISTS = "ITEM_ALREADY_EXISTS"
    INVALID_ITEM_DATA = "INVALID_ITEM_DATA"
    
    # 授權錯誤
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"

# 自定義異常
class AppException(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

# 創建用戶路由器
users_router = APIRouter(
    prefix="/users",
    tags=["users"]
)

# 用戶路由器端點
@users_router.get("/{user_id}")
async def read_user(user_id: int):
    if user_id <= 0:
        raise AppException(
            code=ErrorCode.USER_NOT_FOUND,
            message=f"User with ID {user_id} not found",
            status_code=404
        )
    return {"user_id": user_id, "name": f"User {user_id}"}

# 創建項目路由器
items_router = APIRouter(
    prefix="/items",
    tags=["items"]
)

# 項目路由器端點
@items_router.get("/{item_id}")
async def read_item(item_id: int):
    if item_id <= 0:
        raise AppException(
            code=ErrorCode.ITEM_NOT_FOUND,
            message=f"Item with ID {item_id} not found",
            status_code=404
        )
    return {"item_id": item_id, "name": f"Item {item_id}"}

# 創建主應用
app = FastAPI()

# 應用級別的異常處理器
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    logger.error(f"App exception: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.code,
            "message": exc.message
        }
    )

# 將路由器包含在應用中
app.include_router(users_router)
app.include_router(items_router)
```

在這個例子中，我們定義了一個 `ErrorCode` 枚舉，包含所有可能的錯誤碼，並創建了一個 `AppException` 異常類，用於在應用中拋出標準化的異常。然後，我們在應用級別定義了一個異常處理器，處理所有 `AppException` 異常。

## 5.7 模塊化的異常處理

在大型應用中，您可能希望將異常處理邏輯模塊化，以便在多個路由器之間共享。以下是一種可能的組織方式：

### 項目結構

```
app/
├── main.py                  # 主應用
├── exceptions/
│   ├── __init__.py
│   ├── base.py              # 基礎異常類
│   ├── handlers.py          # 異常處理器
│   └── error_codes.py       # 錯誤碼
├── routers/
│   ├── __init__.py
│   ├── users.py             # 用戶路由器
│   └── items.py             # 項目路由器
└── dependencies/
    ├── __init__.py
    └── auth.py              # 身份驗證依賴項
```

### 實現

```python
# app/exceptions/error_codes.py
from enum import Enum

class ErrorCode(str, Enum):
    # 用戶錯誤
    USER_NOT_FOUND = "USER_NOT_FOUND"
    USER_ALREADY_EXISTS = "USER_ALREADY_EXISTS"
    INVALID_USER_DATA = "INVALID_USER_DATA"
    
    # 項目錯誤
    ITEM_NOT_FOUND = "ITEM_NOT_FOUND"
    ITEM_ALREADY_EXISTS = "ITEM_ALREADY_EXISTS"
    INVALID_ITEM_DATA = "INVALID_ITEM_DATA"
    
    # 授權錯誤
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
```

```python
# app/exceptions/base.py
from .error_codes import ErrorCode

class AppException(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class UserException(AppException):
    """用戶相關異常的基類"""
    pass

class UserNotFoundError(UserException):
    def __init__(self, user_id: int):
        super().__init__(
            code=ErrorCode.USER_NOT_FOUND,
            message=f"User with ID {user_id} not found",
            status_code=404
        )

class ItemException(AppException):
    """項目相關異常的基類"""
    pass

class ItemNotFoundError(ItemException):
    def __init__(self, item_id: int):
        super().__init__(
            code=ErrorCode.ITEM_NOT_FOUND,
            message=f"Item with ID {item_id} not found",
            status_code=404
        )
```

```python
# app/exceptions/handlers.py
from fastapi import Request
from fastapi.responses import JSONResponse
import logging
from .base import AppException, UserException, ItemException

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 應用異常處理器
async def app_exception_handler(request: Request, exc: AppException):
    logger.error(f"App exception: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.code,
            "message": exc.message
        }
    )

# 用戶異常處理器
async def user_exception_handler(request: Request, exc: UserException):
    logger.error(f"User exception: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.code,
            "message": exc.message,
            "entity_type": "user"
        }
    )

# 項目異常處理器
async def item_exception_handler(request: Request, exc: ItemException):
    logger.error(f"Item exception: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.code,
            "message": exc.message,
            "entity_type": "item"
        }
    )
```

```python
# app/routers/users.py
from fastapi import APIRouter
from app.exceptions.base import UserNotFoundError

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.get("/{user_id}")
async def read_user(user_id: int):
    if user_id <= 0:
        raise UserNotFoundError(user_id)
    return {"user_id": user_id, "name": f"User {user_id}"}
```

```python
# app/routers/items.py
from fastapi import APIRouter
from app.exceptions.base import ItemNotFoundError

router = APIRouter(
    prefix="/items",
    tags=["items"]
)

@router.get("/{item_id}")
async def read_item(item_id: int):
    if item_id <= 0:
        raise ItemNotFoundError(item_id)
    return {"item_id": item_id, "name": f"Item {item_id}"}
```

```python
# app/main.py
from fastapi import FastAPI
from app.exceptions.base import AppException, UserException, ItemException
from app.exceptions.handlers import (
    app_exception_handler,
    user_exception_handler,
    item_exception_handler
)
from app.routers import users, items

app = FastAPI()

# 註冊異常處理器
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(UserException, user_exception_handler)
app.add_exception_handler(ItemException, item_exception_handler)

# 包含路由器
app.include_router(users.router)
app.include_router(items.router)
```

這種模塊化的方法使得異常處理邏輯更加清晰和可維護，特別是在大型應用中。

## 5.8 路由器與異常處理的最佳實踐

### 按領域組織路由器

將相關的端點組織到同一個路由器中，並為每個領域定義特定的異常：

```python
# 用戶路由器
users_router = APIRouter(
    prefix="/users",
    tags=["users"]
)

# 項目路由器
items_router = APIRouter(
    prefix="/items",
    tags=["items"]
)

# 訂單路由器
orders_router = APIRouter(
    prefix="/orders",
    tags=["orders"]
)
```

### 使用路由器標籤和描述

為路由器添加標籤和描述，以改善 API 文檔：

```python
users_router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "User not found"}},
    dependencies=[Depends(get_token_header)],
)
```

### 路由器特定的依賴項

為路由器定義特定的依賴項，以實現橫切關注點：

```python
users_router = APIRouter(
    prefix="/users",
    tags=["users"],
    dependencies=[Depends(verify_user_token)]
)

admin_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(verify_admin_token)]
)
```

### 路由器特定的錯誤響應

為每個路由器定義特定的錯誤響應，以提供更詳細的 API 文檔：

```python
users_router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={
        404: {"description": "User not found"},
        400: {"description": "Invalid user data"},
        403: {"description": "Forbidden"}
    }
)
```

### 路由器特定的異常處理器

為每個路由器定義特定的異常處理器，以處理領域特定的異常：

```python
@users_router.exception_handler(UserNotFoundError)
async def user_not_found_exception_handler(request: Request, exc: UserNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": exc.message,
            "user_id": exc.user_id
        }
    )
```

## 5.9 實際案例：電子商務 API

以下是一個電子商務 API 的簡化版本，展示了如何使用 APIRouter 和異常處理：

```python
from fastapi import APIRouter, FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 錯誤碼枚舉
class ErrorCode(str, Enum):
    # 產品錯誤
    PRODUCT_NOT_FOUND = "PRODUCT_NOT_FOUND"
    PRODUCT_OUT_OF_STOCK = "PRODUCT_OUT_OF_STOCK"
    
    # 購物車錯誤
    CART_NOT_FOUND = "CART_NOT_FOUND"
    CART_EMPTY = "CART_EMPTY"
    
    # 訂單錯誤
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    PAYMENT_FAILED = "PAYMENT_FAILED"

# 自定義異常
class AppException(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class ProductException(AppException):
    """產品相關異常的基類"""
    pass

class ProductNotFoundError(ProductException):
    def __init__(self, product_id: int):
        super().__init__(
            code=ErrorCode.PRODUCT_NOT_FOUND,
            message=f"Product with ID {product_id} not found",
            status_code=404
        )

class ProductOutOfStockError(ProductException):
    def __init__(self, product_id: int):
        super().__init__(
            code=ErrorCode.PRODUCT_OUT_OF_STOCK,
            message=f"Product with ID {product_id} is out of stock",
            status_code=400
        )

# 模型
class Product(BaseModel):
    id: int
    name: str
    price: float
    stock: int

class CartItem(BaseModel):
    product_id: int
    quantity: int

class Cart(BaseModel):
    id: int
    items: List[CartItem]

# 產品路由器
products_router = APIRouter(
    prefix="/products",
    tags=["products"]
)

@products_router.get("/", response_model=List[Product])
async def list_products():
    # 模擬產品列表
    return [
        {"id": 1, "name": "Product 1", "price": 10.0, "stock": 5},
        {"id": 2, "name": "Product 2", "price": 20.0, "stock": 10},
        {"id": 3, "name": "Product 3", "price": 30.0, "stock": 0}
    ]

@products_router.get("/{product_id}", response_model=Product)
async def get_product(product_id: int):
    # 模擬產品查詢
    if product_id <= 0 or product_id > 3:
        raise ProductNotFoundError(product_id)
    
    products = {
        1: {"id": 1, "name": "Product 1", "price": 10.0, "stock": 5},
        2: {"id": 2, "name": "Product 2", "price": 20.0, "stock": 10},
        3: {"id": 3, "name": "Product 3", "price": 30.0, "stock": 0}
    }
    
    return products[product_id]

# 購物車路由器
cart_router = APIRouter(
    prefix="/cart",
    tags=["cart"]
)

@cart_router.post("/add/{product_id}")
async def add_to_cart(product_id: int, quantity: int = 1):
    # 模擬添加到購物車
    if product_id <= 0 or product_id > 3:
        raise ProductNotFoundError(product_id)
    
    products = {
        1: {"id": 1, "name": "Product 1", "price": 10.0, "stock": 5},
        2: {"id": 2, "name": "Product 2", "price": 20.0, "stock": 10},
        3: {"id": 3, "name": "Product 3", "price": 30.0, "stock": 0}
    }
    
    if products[product_id]["stock"] == 0:
        raise ProductOutOfStockError(product_id)
    
    if quantity > products[product_id]["stock"]:
        raise AppException(
            code=ErrorCode.PRODUCT_OUT_OF_STOCK,
            message=f"Not enough stock for product {product_id}. Requested: {quantity}, Available: {products[product_id]['stock']}",
            status_code=400
        )
    
    return {"status": "success", "message": f"Added {quantity} of product {product_id} to cart"}

# 創建主應用
app = FastAPI(title="E-Commerce API")

# 應用級別的異常處理器
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    logger.error(f"App exception: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.code,
            "message": exc.message
        }
    )

# 產品異常處理器
@app.exception_handler(ProductException)
async def product_exception_handler(request: Request, exc: ProductException):
    logger.error(f"Product exception: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.code,
            "message": exc.message,
            "entity_type": "product"
        }
    )

# 包含路由器
app.include_router(products_router)
app.include_router(cart_router)
```

在這個例子中，我們創建了一個簡單的電子商務 API，包含產品和購物車路由器。我們定義了特定領域的異常，如 `ProductNotFoundError` 和 `ProductOutOfStockError`，並在應用級別註冊了異常處理器。

## 小結

APIRouter 是組織大型 FastAPI 應用的強大工具，與異常處理結合使用時，可以實現更模塊化、更可維護的代碼：

- **模塊化路由**：將相關端點組織到不同的路由器中，使代碼結構更清晰
- **路由器級別的異常處理**：為特定路由器定義專門的異常處理器，處理領域特定的異常
- **異常處理優先順序**：路由器級別的處理器優先於應用級別的處理器，提供更精細的控制
- **依賴項與異常處理**：使用路由器依賴項實現橫切關注點，如身份驗證和授權
- **標準化錯誤碼**：在所有路由器中使用一致的錯誤碼，提高 API 的一致性
- **模塊化異常定義**：將異常定義和處理邏輯模塊化，便於在多個路由器之間共享
- **領域特定異常**：為每個業務領域定義特定的異常，更準確地反映錯誤情況

通過結合 APIRouter 和全局異常處理，您可以構建結構良好、錯誤處理完善的 FastAPI 應用，提高代碼的可維護性和可擴展性。在下一章中，我們將探討如何在實際項目中實施和測試異常處理策略。

在大型應用中，良好的異常處理策略與模塊化的路由結構相結合，能夠顯著提高代碼質量和開發效率。通過遵循本章介紹的最佳實踐，您可以設計出更健壯、更易於維護的 API，為用戶和開發者提供更好的體驗。