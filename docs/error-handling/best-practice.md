# 8. 異常處理: 最佳實踐

本章將探討 FastAPI 異常處理的最佳實踐，包括設計原則、性能考量、安全性建議和維護策略等，幫助您構建高質量、可維護的異常處理系統。

## 8.1 異常處理設計原則

設計良好的異常處理系統應遵循以下原則：

### 一致性原則

API 的錯誤響應應保持一致的格式和語義，讓客戶端能夠可靠地處理錯誤。

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from enum import Enum
from typing import Optional, Dict, Any
import uuid

# 錯誤碼枚舉
class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"

# 錯誤響應模型
class ErrorResponse(BaseModel):
    status: str = "error"
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: str

app = FastAPI()

# 全局異常處理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # 生成請求 ID
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # 返回一致的錯誤響應
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="An unexpected error occurred",
            details=None,
            request_id=request_id
        ).dict()
    )
```

### 分層原則

將異常處理分為多個層次，每個層次處理特定類型的異常。

```python
# 1. 應用層異常 - 處理業務邏輯錯誤
class AppException(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)

# 2. 基礎設施層異常 - 處理數據庫、緩存等錯誤
class InfrastructureException(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)

# 異常處理器
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request_id=request_id
        ).dict()
    )
```

### 信息隱藏原則

在生產環境中，不應向客戶端暴露敏感的錯誤詳情，如堆棧跟蹤、服務器路徑等。

```python
from fastapi import FastAPI, Request
import logging
import traceback
import os

app = FastAPI()

# 環境配置
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# 全局異常處理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # 生成請求 ID
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # 記錄完整的異常信息
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # 根據環境返回不同級別的錯誤信息
    if DEBUG:
        # 開發環境：返回詳細錯誤信息
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(exc),
                "exception_type": exc.__class__.__name__,
                "traceback": traceback.format_exc().split("\n"),
                "request_id": request_id
            }
        )
    else:
        # 生產環境：返回有限的錯誤信息
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message="An unexpected error occurred",
                details=None,
                request_id=request_id
            ).dict()
        )
```

### 明確性原則

錯誤消息應清晰明確，幫助客戶端理解發生了什麼以及如何解決問題。

```python
# 不好的錯誤消息
@app.get("/users/{user_id}/bad")
async def get_user_bad(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="Error")
    # ...

# 好的錯誤消息
@app.get("/users/{user_id}/good")
async def get_user_good(user_id: str):
    if not user_id:
        raise HTTPException(
            status_code=400, 
            detail="User ID cannot be empty"
        )
    # ...

# 更好的錯誤消息
class UserInputError(AppException):
    def __init__(self, field: str, reason: str, value: Any = None):
        details = {
            "field": field,
            "reason": reason
        }
        if value is not None:
            details["provided_value"] = str(value)
        
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid input for field '{field}': {reason}",
            status_code=400,
            details=details
        )
```

## 8.2 性能考量

異常處理會影響應用的性能，需要考慮以下因素：

### 避免過度使用異常

異常應該用於處理異常情況，而不是正常的控制流程。

```python
# 不好的做法：使用異常進行正常的控制流程
@app.get("/users/{username}")
async def get_user_by_username_bad(username: str):
    try:
        user = await find_user_by_username(username)
        return user
    except UserNotFoundError:
        return {"username": username, "status": "new"}

# 好的做法：使用條件邏輯進行正常的控制流程
@app.get("/users/{username}")
async def get_user_by_username_good(username: str):
    user = await find_user_by_username(username)
    if user is None:
        return {"username": username, "status": "new"}
    return user
```

### 優化異常處理器

異常處理器應該高效，避免執行複雜的邏輯。

```python
# 不好的做法：異常處理器中執行複雜邏輯
@app.exception_handler(HTTPException)
async def http_exception_handler_bad(request: Request, exc: HTTPException):
    # 不必要的複雜處理
    await complex_logging_logic(request, exc)  # 可能導致性能問題
    await notify_admin(exc)  # 同步通知管理員
    
    # 生成響應
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# 好的做法：異常處理器保持簡潔，將複雜邏輯放入背景任務
@app.exception_handler(HTTPException)
async def http_exception_handler_good(request: Request, exc: HTTPException):
    # 記錄基本信息
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # 啟動背景任務處理複雜邏輯
    if exc.status_code >= 500:
        background_tasks = BackgroundTasks()
        background_tasks.add_task(complex_logging_logic, request, exc)
    
    # 快速生成響應
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id}
    )
```

### 使用適當的日誌級別

根據異常的嚴重性使用適當的日誌級別，避免日誌系統成為性能瓶頸。

```python
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 異常處理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # 根據狀態碼使用不同的日誌級別
    if exc.status_code >= 500:
        # 服務器錯誤：使用 ERROR 級別
        logger.error(f"Server error: {exc.detail}", exc_info=True)
    elif exc.status_code >= 400:
        # 客戶端錯誤：使用 WARNING 級別
        logger.warning(f"Client error: {exc.detail}")
    else:
        # 其他：使用 INFO 級別
        logger.info(f"HTTP exception: {exc.detail}")
    
    # 返回響應
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
```

## 8.3 安全性建議

異常處理與應用安全密切相關，需要注意以下安全建議：

### 避免暴露敏感信息

錯誤響應不應包含敏感信息，如密碼、API 密鑰、內部路徑等。

```python
# 不安全的做法：暴露敏感信息
@app.post("/login")
async def login_unsafe(username: str, password: str):
    try:
        # 嘗試登錄
        user = authenticate(username, password)
        return {"token": generate_token(user)}
    except Exception as e:
        # 不安全：可能暴露密碼或其他敏感信息
        raise HTTPException(
            status_code=400,
            detail=f"Login failed: {str(e)}, username={username}, password={password}"
        )

# 安全的做法：不暴露敏感信息
@app.post("/login")
async def login_safe(username: str, password: str):
    try:
        # 嘗試登錄
        user = authenticate(username, password)
        return {"token": generate_token(user)}
    except AuthenticationError:
        # 安全：提供有限的錯誤信息
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )
    except Exception as e:
        # 記錄詳細錯誤，但不返回給客戶端
        logging.error(f"Login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during login"
        )
```

### 防止信息洩露

錯誤響應不應透露系統的內部結構或實現細節。

```python
# 不安全的做法：暴露系統細節
@app.exception_handler(Exception)
async def global_exception_handler_unsafe(request: Request, exc: Exception):
    # 不安全：暴露了堆棧跟蹤和服務器路徑
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "server_path": os.path.abspath(__file__),
            "python_version": sys.version
        }
    )

# 安全的做法：隱藏系統細節
@app.exception_handler(Exception)
async def global_exception_handler_safe(request: Request, exc: Exception):
    # 記錄詳細信息，但不返回給客戶端
    error_id = str(uuid.uuid4())
    logging.error(f"Error ID: {error_id}", exc_info=True)
    
    # 返回有限的錯誤信息
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "error_id": error_id
        }
    )
```

### 防止枚舉攻擊

避免通過錯誤響應洩露資源是否存在的信息。

```python
# 不安全的做法：可能導致枚舉攻擊
@app.get("/users/{user_id}")
async def get_user_unsafe(user_id: str):
    user = await find_user(user_id)
    if user is None:
        # 不安全：確認了特定 ID 的用戶不存在
        raise HTTPException(
            status_code=404,
            detail=f"User with ID {user_id} does not exist"
        )
    return user

# 安全的做法：使用一致的錯誤消息
@app.get("/users/{user_id}")
async def get_user_safe(user_id: str):
    user = await find_user(user_id)
    if user is None:
        # 安全：使用一致的錯誤消息
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    return user
```

### 安全的錯誤處理中間件

實現一個安全的錯誤處理中間件，統一處理安全相關的異常。

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.base import BaseHTTPMiddleware
import uuid

app = FastAPI()

# 安全錯誤處理中間件
class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 生成請求 ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 添加安全相關的響應頭
        headers = {
            "X-Request-ID": request_id,
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY"
        }
        
        try:
            # 處理請求
            response = await call_next(request)
            
            # 添加安全頭到響應
            for key, value in headers.items():
                response.headers[key] = value
            
            return response
        
        except Exception as exc:
            # 處理未捕獲的異常
            logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
            
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "An unexpected error occurred",
                    "request_id": request_id
                },
                headers=headers
            )

# 添加中間件
app.add_middleware(SecurityMiddleware)
```

## 8.4 可維護性策略

設計易於維護的異常處理系統，可以提高開發效率和代碼質量。

### 集中式異常定義

將所有異常類型集中定義在一個模塊中，便於管理和維護。

```python
# exceptions.py
from enum import Enum
from typing import Optional, Dict, Any

# 錯誤碼枚舉
class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"

# 基礎異常類
class AppException(Exception):
    def __init__(
        self, 
        code: ErrorCode, 
        message: str, 
        status_code: int = 400, 
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)

# 驗證異常
class ValidationError(AppException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            status_code=422,
            details=details
        )

# 認證異常
class AuthenticationError(AppException):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            code=ErrorCode.AUTHENTICATION_ERROR,
            message=message,
            status_code=401
        )

# 資源不存在異常
class ResourceNotFoundError(AppException):
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_FOUND,
            message=f"{resource_type} not found",
            status_code=404,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )
```

### 模塊化異常處理

將異常處理邏輯模塊化，便於復用和測試。

```python
# handlers.py
from fastapi import Request
from fastapi.responses import JSONResponse
import logging
import uuid
from .exceptions import AppException, ErrorCode

# 錯誤響應模型
class ErrorResponse:
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        self.status = "error"
        self.code = code
        self.message = message
        self.details = details
        self.request_id = request_id or str(uuid.uuid4())
    
    def dict(self):
        return {
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "request_id": self.request_id
        }

# 應用異常處理器
async def app_exception_handler(request: Request, exc: AppException):
    # 獲取請求 ID
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # 記錄錯誤
    if exc.status_code >= 500:
        logging.error(f"Application error: {exc.message}", exc_info=True)
    
    # 返回錯誤響應
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request_id=request_id
        ).dict()
    )

# 註冊異常處理器
def register_exception_handlers(app: FastAPI):
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
```

### 使用工廠模式創建異常

使用工廠模式創建異常，可以簡化異常的創建和管理。

```python
# exception_factory.py
from typing import Optional, Dict, Any
from .exceptions import (
    ResourceNotFoundError, ValidationError
)

class ExceptionFactory:
    @staticmethod
    def resource_not_found(resource_type: str, resource_id: str) -> ResourceNotFoundError:
        return ResourceNotFoundError(resource_type, resource_id)
    
    @staticmethod
    def validation_error(message: str, details: Optional[Dict[str, Any]] = None) -> ValidationError:
        return ValidationError(message, details)

# 使用工廠創建異常
@app.get("/users/{user_id}")
async def get_user(user_id: str):
    user = await find_user(user_id)
    if user is None:
        raise ExceptionFactory.resource_not_found("User", user_id)
    return user
```

## 8.5 異常處理測試策略

測試異常處理邏輯是確保應用穩定性的關鍵部分。

```python
# test_exceptions.py
import pytest
from fastapi.testclient import TestClient
from .main import app
from .exceptions import ResourceNotFoundError

client = TestClient(app)

def test_resource_not_found_exception():
    # 測試資源不存在異常
    response = client.get("/users/999")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == "RESOURCE_NOT_FOUND"
    assert "User not found" in data["message"]
    assert data["details"]["resource_id"] == "999"

def test_validation_error():
    # 測試驗證錯誤
    response = client.post("/users/", json={"email": "invalid-email"})
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == "VALIDATION_ERROR"
```

## 8.6 異常處理文檔

為 API 的異常處理提供清晰的文檔，幫助客戶端開發者理解和處理錯誤。

```python
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi

app = FastAPI()

# 自定義 OpenAPI 模式，添加錯誤響應文檔
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="My API",
        version="1.0.0",
        description="API with detailed error responses",
        routes=app.routes,
    )
    
    # 添加通用錯誤響應組件
    openapi_schema["components"]["schemas"]["HTTPValidationError"] = {
        "properties": {
            "status": {"type": "string", "example": "error"},
            "code": {"type": "string", "example": "VALIDATION_ERROR"},
            "message": {"type": "string"},
            "details": {"type": "object"},
            "request_id": {"type": "string", "format": "uuid"}
        },
        "type": "object"
    }
    
    # 為每個路徑添加錯誤響應
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            # 添加 422 驗證錯誤
            openapi_schema["paths"][path][method]["responses"]["422"] = {
                "description": "Validation Error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/HTTPValidationError"}
                    }
                }
            }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

## 8.7 異常監控與分析

實施異常監控和分析，以識別和解決系統中的問題。

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import time
import uuid
import logging
from collections import Counter

app = FastAPI()

# 異常計數器
error_counter = Counter()

# 中間件：記錄異常
@app.middleware("http")
async def log_exceptions_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        # 記錄異常
        error_type = exc.__class__.__name__
        error_counter[error_type] += 1
        
        # 記錄詳細信息
        logging.error(
            f"Request {request_id} failed with {error_type}: {str(exc)}",
            exc_info=True,
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "processing_time": time.time() - start_time
            }
        )
        
        # 重新拋出異常，讓異常處理器處理
        raise

# 監控端點：獲取異常統計
@app.get("/admin/errors/stats")
async def get_error_stats():
    return {
        "total_errors": sum(error_counter.values()),
        "errors_by_type": dict(error_counter)
    }
```

## 8.8 總結

良好的異常處理是構建可靠、安全和用戶友好的 API 的關鍵。通過遵循本章介紹的最佳實踐，您可以設計出高質量的異常處理系統，提高應用的健壯性和可維護性。

關鍵要點：

1. **一致性**: 保持錯誤響應的一致格式和語義
2. **分層**: 將異常處理分為多個層次
3. **安全性**: 避免暴露敏感信息和系統細節
4. **明確性**: 提供清晰、有用的錯誤消息
5. **性能**: 優化異常處理以避免性能問題
6. **可維護性**: 使用模塊化和集中式異常定義
7. **測試**: 全面測試異常處理邏輯
8. **文檔**: 提供詳細的錯誤響應文檔
9. **監控**: 實施異常監控和分析

通過綜合應用這些最佳實踐，您可以構建出既健壯又易於維護的異常處理系統，為用戶提供良好的體驗，同時保護您的應用免受潛在的安全威脅。