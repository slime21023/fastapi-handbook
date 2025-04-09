# 自定義異常處理

在實際應用開發中，FastAPI 的內建異常處理機制雖然強大，但往往無法完全滿足複雜業務邏輯的需求。自定義異常處理允許您根據應用的特定需求，創建更貼合業務場景的錯誤處理機制。

## 3.1 自定義異常類

自定義異常類是處理特定業務邏輯錯誤的基礎。通過繼承 Python 的 `Exception` 類或其子類，您可以創建專門用於處理特定錯誤情況的異常類型。

### 基本自定義異常

```python
class ItemNotFoundError(Exception):
    def __init__(self, item_id: int, message: str = None):
        self.item_id = item_id
        self.message = message or f"Item with ID {item_id} not found"
        super().__init__(self.message)

class InsufficientFundsError(Exception):
    def __init__(self, account_id: str, required: float, available: float):
        self.account_id = account_id
        self.required = required
        self.available = available
        self.message = f"Account {account_id} has insufficient funds: required {required}, available {available}"
        super().__init__(self.message)
```

### 異常層次結構

為了更好地組織異常，可以創建一個異常層次結構：

```python
# 基礎異常類
class AppException(Exception):
    """基礎應用異常類"""
    def __init__(self, message: str = "An application error occurred"):
        self.message = message
        super().__init__(self.message)

# 資源異常
class ResourceException(AppException):
    """資源相關的異常基類"""
    pass

class ResourceNotFoundError(ResourceException):
    """資源未找到異常"""
    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        message = f"{resource_type} with ID {resource_id} not found"
        super().__init__(message)

class ResourceAlreadyExistsError(ResourceException):
    """資源已存在異常"""
    def __init__(self, resource_type: str, identifier: str):
        self.resource_type = resource_type
        self.identifier = identifier
        message = f"{resource_type} with identifier {identifier} already exists"
        super().__init__(message)

# 權限異常
class PermissionException(AppException):
    """權限相關的異常基類"""
    pass

class InsufficientPermissionsError(PermissionException):
    """權限不足異常"""
    def __init__(self, user_id: str, required_permission: str):
        self.user_id = user_id
        self.required_permission = required_permission
        message = f"User {user_id} does not have the required permission: {required_permission}"
        super().__init__(message)
```

這種層次結構使得異常處理更加靈活，您可以選擇捕獲特定的異常或更通用的基類異常。

| 異常類型 | 說明 | 使用場景 |
|---------|------|---------|
| `AppException` | 基礎應用異常類 | 捕獲所有應用級異常 |
| `ResourceException` | 資源相關異常基類 | 捕獲所有資源相關異常 |
| `ResourceNotFoundError` | 資源未找到異常 | 當請求的資源不存在時 |
| `ResourceAlreadyExistsError` | 資源已存在異常 | 當嘗試創建已存在的資源時 |
| `PermissionException` | 權限相關異常基類 | 捕獲所有權限相關異常 |
| `InsufficientPermissionsError` | 權限不足異常 | 當用戶沒有執行操作的權限時 |

## 3.2 註冊異常處理器

在 FastAPI 中，您可以使用 `@app.exception_handler()` 裝飾器註冊自定義異常處理器，將自定義異常轉換為適當的 HTTP 響應。

### 基本異常處理器

```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

app = FastAPI()

class ItemNotFoundError(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id

@app.exception_handler(ItemNotFoundError)
async def item_not_found_exception_handler(request: Request, exc: ItemNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "status": "error",
            "message": f"Item with ID {exc.item_id} not found"
        }
    )

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 0:
        raise ItemNotFoundError(item_id)
    return {"item_id": item_id, "name": f"Item {item_id}"}
```

### 為異常層次結構註冊處理器

您可以為異常層次結構中的不同級別註冊處理器，從而實現更靈活的錯誤處理：

```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

app = FastAPI()

# 基礎異常類
class AppException(Exception):
    """基礎應用異常類"""
    def __init__(self, message: str = "An application error occurred"):
        self.message = message
        super().__init__(self.message)

# 資源異常
class ResourceException(AppException):
    """資源相關的異常基類"""
    pass

class ResourceNotFoundError(ResourceException):
    """資源未找到異常"""
    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.message = f"{resource_type} with ID {resource_id} not found"
        super().__init__(self.message)

# 基礎異常處理器
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": exc.message
        }
    )

# 資源異常處理器
@app.exception_handler(ResourceException)
async def resource_exception_handler(request: Request, exc: ResourceException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "message": exc.message
        }
    )

# 資源未找到異常處理器
@app.exception_handler(ResourceNotFoundError)
async def resource_not_found_exception_handler(request: Request, exc: ResourceNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "status": "error",
            "message": exc.message,
            "resource_type": exc.resource_type,
            "resource_id": exc.resource_id
        }
    )

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    if user_id == "0":
        raise ResourceNotFoundError(resource_type="User", resource_id=user_id)
    return {"user_id": user_id, "name": f"User {user_id}"}
```

在上面的例子中，如果拋出 `ResourceNotFoundError`，將由 `resource_not_found_exception_handler` 處理；如果拋出其他 `ResourceException` 子類，將由 `resource_exception_handler` 處理；如果拋出其他 `AppException` 子類，將由 `app_exception_handler` 處理。

## 3.3 結構化錯誤響應

為了提供一致的 API 錯誤響應，可以定義一個標準的錯誤響應模型：

```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional, List

app = FastAPI()

# 錯誤響應模型
class ErrorDetail(BaseModel):
    loc: Optional[List[str]] = None
    msg: str
    type: str

class ErrorResponse(BaseModel):
    status: str = "error"
    code: int
    message: str
    details: Optional[List[ErrorDetail]] = None
    path: Optional[str] = None

# 自定義異常
class ValidationError(Exception):
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(self.message)

# 異常處理器
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    error_detail = ErrorDetail(
        loc=[exc.field],
        msg=exc.message,
        type="validation_error"
    )
    
    error_response = ErrorResponse(
        code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="Validation error",
        details=[error_detail],
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.dict()
    )

@app.post("/users/")
async def create_user(name: str, age: int):
    if len(name) < 3:
        raise ValidationError(field="name", message="Name must be at least 3 characters long")
    if age < 18:
        raise ValidationError(field="age", message="Age must be at least 18")
    return {"name": name, "age": age}
```

這將產生如下格式的錯誤響應：

```json
{
  "status": "error",
  "code": 422,
  "message": "Validation error",
  "details": [
    {
      "loc": ["name"],
      "msg": "Name must be at least 3 characters long",
      "type": "validation_error"
    }
  ],
  "path": "/users/"
}
```

## 3.4 自定義異常與業務邏輯

自定義異常特別適合處理業務邏輯錯誤，這些錯誤通常與應用的特定領域相關。

### 業務邏輯異常示例

```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from datetime import datetime
from enum import Enum

app = FastAPI()

# 訂單狀態枚舉
class OrderStatus(str, Enum):
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

# 業務邏輯異常
class OrderError(Exception):
    """訂單相關異常基類"""
    def __init__(self, order_id: str, message: str):
        self.order_id = order_id
        self.message = message
        super().__init__(self.message)

class OrderNotFoundError(OrderError):
    """訂單未找到異常"""
    def __init__(self, order_id: str):
        super().__init__(order_id, f"Order with ID {order_id} not found")

class InvalidOrderStatusTransitionError(OrderError):
    """訂單狀態轉換無效異常"""
    def __init__(self, order_id: str, current_status: OrderStatus, target_status: OrderStatus):
        self.current_status = current_status
        self.target_status = target_status
        message = f"Cannot transition order {order_id} from {current_status} to {target_status}"
        super().__init__(order_id, message)

class OrderAlreadyCancelledError(OrderError):
    """訂單已取消異常"""
    def __init__(self, order_id: str, cancelled_at: datetime):
        self.cancelled_at = cancelled_at
        message = f"Order {order_id} was already cancelled at {cancelled_at}"
        super().__init__(order_id, message)

# 異常處理器
@app.exception_handler(OrderError)
async def order_exception_handler(request: Request, exc: OrderError):
    status_code = status.HTTP_400_BAD_REQUEST
    
    # 根據異常類型設置不同的狀態碼
    if isinstance(exc, OrderNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": exc.message,
            "order_id": exc.order_id,
            "error_type": exc.__class__.__name__
        }
    )

# API 端點
@app.put("/orders/{order_id}/status")
async def update_order_status(order_id: str, new_status: OrderStatus):
    # 模擬數據庫查詢
    if order_id == "not-found":
        raise OrderNotFoundError(order_id)
    
    # 模擬訂單狀態
    current_status = OrderStatus.PAID
    
    # 業務邏輯檢查
    if current_status == OrderStatus.CANCELLED:
        raise OrderAlreadyCancelledError(
            order_id=order_id,
            cancelled_at=datetime.now()
        )
    
    # 檢查狀態轉換是否有效
    valid_transitions = {
        OrderStatus.PENDING: [OrderStatus.PAID, OrderStatus.CANCELLED],
        OrderStatus.PAID: [OrderStatus.SHIPPED, OrderStatus.CANCELLED],
        OrderStatus.SHIPPED: [OrderStatus.DELIVERED, OrderStatus.CANCELLED],
        OrderStatus.DELIVERED: [],
        OrderStatus.CANCELLED: []
    }
    
    if new_status not in valid_transitions[current_status]:
        raise InvalidOrderStatusTransitionError(
            order_id=order_id,
            current_status=current_status,
            target_status=new_status
        )
    
    # 更新訂單狀態
    return {
        "order_id": order_id,
        "previous_status": current_status,
        "current_status": new_status,
        "updated_at": datetime.now().isoformat()
    }
```

## 3.5 異常處理的最佳實踐

### 異常命名約定

| 約定 | 示例 | 說明 |
|------|------|------|
| 使用 `Error` 後綴 | `ResourceNotFoundError` | 清晰地表明這是一個錯誤類型 |
| 使用描述性名稱 | `InsufficientFundsError` | 名稱應該清晰描述錯誤情況 |
| 使用層次結構 | `PaymentError` → `PaymentProcessingError` | 使用繼承創建有意義的層次結構 |

### 異常處理的最佳實踐

| 最佳實踐 | 說明 |
|---------|------|
| **提供有用的錯誤信息** | 錯誤信息應該清晰、具體，並幫助用戶理解和解決問題 |
| **包含相關上下文** | 在異常中包含相關的上下文信息，如資源 ID、用戶 ID 等 |
| **使用適當的 HTTP 狀態碼** | 確保每種異常類型映射到適當的 HTTP 狀態碼 |
| **保持一致的響應格式** | 所有錯誤響應應遵循一致的格式，便於客戶端處理 |
| **避免洩露敏感信息** | 確保錯誤信息不包含敏感數據，如密碼、內部路徑等 |
| **記錄異常** | 在返回錯誤響應前記錄異常，便於調試和監控 |
| **區分客戶端和服務器錯誤** | 使用 4xx 狀態碼表示客戶端錯誤，5xx 表示服務器錯誤 |

### 完整示例：綜合自定義異常處理

```python
from fastapi import FastAPI, Request, status, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import logging
from datetime import datetime

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 基礎異常類
class AppException(Exception):
    """基礎應用異常類"""
    def __init__(
        self, 
        message: str = "An application error occurred",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "internal_error",
        details: Optional[Dict] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

# 資源異常
class ResourceException(AppException):
    """資源相關的異常基類"""
    def __init__(
        self, 
        message: str, 
        status_code: int = status.HTTP_400_BAD_REQUEST,
        error_code: str = "resource_error",
        details: Optional[Dict] = None
    ):
        super().__init__(message, status_code, error_code, details)

class ResourceNotFoundError(ResourceException):
    """資源未找到異常"""
    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        message = f"{resource_type} with ID {resource_id} not found"
        details = {"resource_type": resource_type, "resource_id": resource_id}
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="resource_not_found",
            details=details
        )

# 權限異常
class PermissionException(AppException):
    """權限相關的異常基類"""
    def __init__(
        self, 
        message: str, 
        status_code: int = status.HTTP_403_FORBIDDEN,
        error_code: str = "permission_error",
        details: Optional[Dict] = None
    ):
        super().__init__(message, status_code, error_code, details)

class InsufficientPermissionsError(PermissionException):
    """權限不足異常"""
    def __init__(self, user_id: str, required_permission: str):
        self.user_id = user_id
        self.required_permission = required_permission
        message = f"User {user_id} does not have the required permission: {required_permission}"
        details = {"user_id": user_id, "required_permission": required_permission}
        super().__init__(
            message=message,
            error_code="insufficient_permissions",
            details=details
        )

# 驗證異常
class ValidationException(AppException):
    """驗證相關的異常基類"""
    def __init__(
        self, 
        message: str, 
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code: str = "validation_error",
        details: Optional[Dict] = None
    ):
        super().__init__(message, status_code, error_code, details)

class InvalidInputError(ValidationException):
    """輸入無效異常"""
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        message = f"Invalid input for field '{field}': {reason}"
        details = {"field": field, "reason": reason}
        super().__init__(
            message=message,
            error_code="invalid_input",
            details=details
        )

# 通用異常處理器
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    # 記錄異常
    logger.error(
        f"Exception occurred: {exc.__class__.__name__}, "
        f"Message: {exc.message}, "
        f"Path: {request.url.path}"
    )
    
    response_content = {
        "status": "error",
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": datetime.now().isoformat()
        },
        "path": request.url.path
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_content
    )

# 模擬認證依賴
def get_current_user(user_id: str = None):
    if not user_id:
        raise AppException(
            message="Authentication required",
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="authentication_required"
        )
    return {"user_id": user_id, "name": f"User {user_id}"}

# API 端點
@app.get("/users/{user_id}")
async def read_user(user_id: str):
    if user_id == "0":
        raise ResourceNotFoundError(resource_type="User", resource_id=user_id)
    return {"user_id": user_id, "name": f"User {user_id}"}

@app.post("/items/")
async def create_item(name: str, price: float, current_user: Dict = Depends(get_current_user)):
    if len(name) < 3:
        raise InvalidInputError(field="name", reason="Must be at least 3 characters long")
    if price <= 0:
        raise InvalidInputError(field="price", reason="Must be greater than 0")
    return {
        "id": "123",
        "name": name,
        "price": price,
        "created_by": current_user["user_id"]
    }

@app.delete("/admin/users/{user_id}")
async def delete_user(user_id: str, current_user: Dict = Depends(get_current_user)):
    # 檢查權限
    if current_user["user_id"] != "admin":
        raise InsufficientPermissionsError(
            user_id=current_user["user_id"],
            required_permission="admin"
        )
    
    # 檢查用戶是否存在
    if user_id == "0":
        raise ResourceNotFoundError(resource_type="User", resource_id=user_id)
    
    return {"status": "success", "message": f"User {user_id} deleted"}
```

## 3.6 自定義異常與依賴項

自定義異常可以與 FastAPI 的依賴項系統結合使用，實現更複雜的錯誤處理邏輯：

```python
from fastapi import FastAPI, Depends, Header
from typing import Optional

app = FastAPI()

# 自定義異常
class AuthenticationError(Exception):
    def __init__(self, message: str = "Authentication failed"):
        self.message = message
        super().__init__(self.message)

class AuthorizationError(Exception):
    def __init__(self, message: str = "Authorization failed"):
        self.message = message
        super().__init__(self.message)

# 依賴項
def verify_token(token: Optional[str] = Header(None)):
    if not token:
        raise AuthenticationError("Token is missing")
    if token != "valid_token":
        raise AuthenticationError("Invalid token")
    return {"user_id": "123"}

def verify_admin(user = Depends(verify_token)):
    if user["user_id"] != "admin":
        raise AuthorizationError("Admin privileges required")
    return user

# 異常處理器
@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"status": "error", "message": exc.message}
    )

@app.exception_handler(AuthorizationError)
async def authorization_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"status": "error", "message": exc.message}
    )

# API 端點
@app.get("/users/me", dependencies=[Depends(verify_token)])
async def read_user_me():
    return {"user_id": "123", "name": "Current User"}

@app.get("/admin", dependencies=[Depends(verify_admin)])
async def admin_panel():
    return {"status": "success", "message": "Welcome to admin panel"}
```

## 3.7 自定義異常與請求驗證

自定義異常可以與 Pydantic 模型結合，實現更強大的請求驗證：

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List

app = FastAPI()

# 自定義驗證異常
class ProductValidationError(Exception):
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

# Pydantic 模型
class Product(BaseModel):
    name: str
    price: float
    categories: List[str]
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if len(v) < 3:
            raise ProductValidationError("name", "Product name must be at least 3 characters long")
        if len(v) > 50:
            raise ProductValidationError("name", "Product name must be at most 50 characters long")
        return v
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ProductValidationError("price", "Product price must be positive")
        return v
    
    @validator('categories')
    def categories_must_not_be_empty(cls, v):
        if not v:
            raise ProductValidationError("categories", "Product must have at least one category")
        return v

# 異常處理器
@app.exception_handler(ProductValidationError)
async def product_validation_error_handler(request: Request, exc: ProductValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Product validation error",
            "field": exc.field,
            "detail": exc.message
        }
    )

# API 端點
@app.post("/products/")
async def create_product(product: Product):
    return {"status": "success", "product": product.dict()}
```

## 小結

自定義異常處理是構建健壯、易於維護的 FastAPI 應用的關鍵部分：

- **自定義異常類**：創建特定於業務邏輯的異常類型，包含相關上下文信息
- **異常層次結構**：使用繼承創建有意義的異常層次結構，便於組織和處理
- **註冊異常處理器**：使用 `@app.exception_handler()` 裝飾器將自定義異常映射到 HTTP 響應
- **結構化錯誤響應**：提供一致、信息豐富的錯誤響應格式
- **與業務邏輯結合**：使用自定義異常表達業務規則和約束
- **與依賴項結合**：在依賴項中使用自定義異常進行身份驗證和授權
- **與請求驗證結合**：在 Pydantic 模型中使用自定義異常進行高級數據驗證

通過實施這些實踐，您可以創建具有清晰、一致的錯誤處理機制的 API，提供良好的開發者體驗，並簡化調試和維護。
