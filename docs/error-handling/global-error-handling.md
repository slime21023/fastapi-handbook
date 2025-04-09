# 4. 全局異常處理器

在 FastAPI 應用中，全局異常處理器是確保所有異常都得到適當處理的最後一道防線。無論是預期的異常還是未預期的異常，全局處理器都能捕獲它們，並將其轉換為一致的 HTTP 響應。

## 4.1 全局異常處理的重要性

全局異常處理在 API 開發中扮演著關鍵角色，具有以下優勢：

| 優勢 | 說明 |
|------|------|
| **防止應用崩潰** | 捕獲所有未處理的異常，確保應用不會因為意外錯誤而崩潰 |
| **提供一致的錯誤響應** | 確保所有錯誤都以一致的格式返回給客戶端 |
| **簡化錯誤處理** | 集中處理常見錯誤，減少在每個端點中重複編寫錯誤處理代碼 |
| **改善調試體驗** | 提供詳細的錯誤信息，便於開發和調試 |
| **增強安全性** | 在生產環境中隱藏敏感的錯誤詳情，防止信息洩露 |
| **支持日誌記錄和監控** | 集中記錄所有錯誤，便於監控和分析 |

## 4.2 處理內建異常

FastAPI 提供了多種方法來處理內建異常，如 `RequestValidationError`、`HTTPException` 等。

### 處理 RequestValidationError

`RequestValidationError` 是當請求數據不符合 Pydantic 模型定義時拋出的異常：

```python
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    
    error_messages = []
    for error in exc.errors():
        error_messages.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": error_messages,
            "path": request.url.path
        }
    )
```

### 處理 HTTPException

`HTTPException` 是 FastAPI 中最常用的異常類型：

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import logging

app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    
    content = {
        "status": "error",
        "message": exc.detail,
        "path": request.url.path
    }
    
    headers = getattr(exc, "headers", None)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=headers
    )
```

## 4.3 處理未捕獲的異常

捕獲所有未處理的異常是全局異常處理的關鍵部分：

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
import traceback
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

app = FastAPI()

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unhandled error occurred: {str(exc)}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    
    if DEBUG:
        # 開發環境：返回詳細錯誤信息
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(exc),
                "traceback": traceback.format_exc().split("\n"),
                "path": request.url.path
            }
        )
    else:
        # 生產環境：返回通用錯誤信息
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "An unexpected error occurred. Please try again later.",
                "path": request.url.path
            }
        )
```

## 4.4 自定義全局異常處理中間件

除了使用 `@app.exception_handler()` 裝飾器外，您還可以使用中間件來實現全局異常處理：

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import traceback
import time

app = FastAPI()

class ExceptionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            error_msg = f"Unhandled error occurred: {str(exc)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": "error",
                    "message": "An unexpected error occurred.",
                    "path": request.url.path,
                    "request_id": request.headers.get("X-Request-ID", "unknown"),
                    "timestamp": time.time(),
                    "processing_time": time.time() - start_time
                }
            )

app.add_middleware(ExceptionMiddleware)
```

## 4.5 全局異常處理的最佳實踐

### 標準化錯誤響應格式

定義一個標準的錯誤響應格式，確保所有錯誤響應都遵循相同的結構：

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

app = FastAPI()

# 標準錯誤響應模型
class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str
    type: str

class ErrorResponse(BaseModel):
    status: str = "error"
    code: int
    message: str
    details: Optional[List[ErrorDetail]] = None
    path: str
    timestamp: str
    request_id: Optional[str] = None

# 創建標準錯誤響應
def create_error_response(
    status_code: int,
    message: str,
    request: Request,
    details: Optional[List[ErrorDetail]] = None
) -> JSONResponse:
    response = ErrorResponse(
        code=status_code,
        message=message,
        details=details,
        path=request.url.path,
        timestamp=datetime.now().isoformat(),
        request_id=request.headers.get("X-Request-ID")
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response.dict()
    )

# 使用標準錯誤響應
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="An unexpected error occurred",
        request=request
    )
```

### 全面的異常處理策略

實現一個全面的異常處理策略，處理各種可能的異常情況：

```python
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError
import logging
import traceback
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

app = FastAPI()

# 標準錯誤響應函數
def create_error_response(
    status_code: int,
    message: str,
    request: Request,
    details=None,
    exception_type=None,
    traceback_info=None
) -> JSONResponse:
    content = {
        "status": "error",
        "code": status_code,
        "message": message,
        "path": request.url.path,
        "timestamp": datetime.now().isoformat(),
        "request_id": request.headers.get("X-Request-ID")
    }
    
    if details:
        content["details"] = details
    
    if DEBUG and exception_type:
        content["exception_type"] = exception_type
    
    if DEBUG and traceback_info:
        content["traceback"] = traceback_info
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )

# 處理 RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    
    details = []
    for error in exc.errors():
        details.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="Validation error",
        request=request,
        details=details,
        exception_type="RequestValidationError"
    )

# 處理 HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    
    response = create_error_response(
        status_code=exc.status_code,
        message=exc.detail,
        request=request,
        exception_type="HTTPException"
    )
    
    if hasattr(exc, "headers") and exc.headers:
        for key, value in exc.headers.items():
            response.headers[key] = value
    
    return response

# 處理所有未捕獲的異常
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="An unexpected error occurred" if not DEBUG else str(exc),
        request=request,
        exception_type=exc.__class__.__name__ if DEBUG else None,
        traceback_info=traceback.format_exc().split("\n") if DEBUG else None
    )
```

## 4.6 整合日誌記錄與監控

全局異常處理器是集中記錄和監控錯誤的理想場所：

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
import traceback
import json
import time
from datetime import datetime
import uuid

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加文件處理器
file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)
logger.addHandler(file_handler)

app = FastAPI()

# 生成請求 ID
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    start_time = time.time()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        # 記錄詳細的錯誤信息
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "client_host": request.client.host if request.client else "unknown",
            "exception_type": exc.__class__.__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc()
        }
        
        # 記錄結構化錯誤信息
        logger.error(f"Request error: {json.dumps(error_data)}")
        
        # 返回錯誤響應
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "An unexpected error occurred",
                "request_id": request_id
            }
        )
```

## 4.7 高級全局異常處理技術

### 按環境定制錯誤響應

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
import os
import traceback

# 環境配置
ENV = os.getenv("ENVIRONMENT", "development")

app = FastAPI()

@app.exception_handler(Exception)
async def environment_aware_exception_handler(request: Request, exc: Exception):
    # 記錄錯誤
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    # 根據環境定制錯誤響應
    if ENV == "development":
        # 開發環境：返回詳細信息
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(exc),
                "exception_type": exc.__class__.__name__,
                "traceback": traceback.format_exc().split("\n"),
                "path": request.url.path
            }
        )
    elif ENV == "testing":
        # 測試環境：返回中等詳細程度的信息
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(exc),
                "exception_type": exc.__class__.__name__,
                "path": request.url.path
            }
        )
    else:
        # 生產環境：返回最少信息
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "An unexpected error occurred. Please try again later.",
                "path": request.url.path
            }
        )
```

### 異常分類與處理

將異常分類並根據類別進行不同的處理：

```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import logging
import traceback

app = FastAPI()

# 異常分類
def classify_exception(exc: Exception) -> tuple:
    """將異常分類並返回適當的狀態碼和錯誤信息"""
    # 數據庫相關異常
    if "sqlalchemy" in exc.__class__.__module__:
        return status.HTTP_503_SERVICE_UNAVAILABLE, "Database error occurred"
    
    # 網絡相關異常
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return status.HTTP_503_SERVICE_UNAVAILABLE, "Network error occurred"
    
    # IO 相關異常
    if isinstance(exc, (IOError, FileNotFoundError)):
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "File system error occurred"
    
    # JSON 解析錯誤
    if isinstance(exc, json.JSONDecodeError):
        return status.HTTP_400_BAD_REQUEST, "Invalid JSON format"
    
    # 默認為服務器錯誤
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "An unexpected error occurred"

@app.exception_handler(Exception)
async def classified_exception_handler(request: Request, exc: Exception):
    # 記錄錯誤
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    # 分類異常
    status_code, error_message = classify_exception(exc)
    
    # 返回適當的響應
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": error_message,
            "path": request.url.path
        }
    )
```

## 4.8 整合錯誤報告服務

將未捕獲的異常發送到錯誤報告服務，如 Sentry：

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
import traceback
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

# 配置 Sentry
sentry_sdk.init(
    dsn="https://your-sentry-dsn@sentry.io/project",
    traces_sample_rate=1.0,
    environment=os.getenv("ENVIRONMENT", "development")
)

app = FastAPI()

@app.exception_handler(Exception)
async def sentry_exception_handler(request: Request, exc: Exception):
    # 記錄錯誤
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    # Sentry 會自動捕獲異常，但我們可以添加額外上下文
    with sentry_sdk.push_scope() as scope:
        scope.set_context("request", {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers)
        })
        sentry_sdk.capture_exception(exc)
    
    # 返回錯誤響應
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An unexpected error occurred. Our team has been notified.",
            "error_id": sentry_sdk.last_event_id(),
            "path": request.url.path
        }
    )

# 添加 Sentry 中間件
app.add_middleware(SentryAsgiMiddleware)
```

## 4.9 實用的全局異常處理示例

以下是一個實用的全局異常處理示例，適合大多數 FastAPI 應用：

```python
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
import traceback
import os
from datetime import datetime

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境配置
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

app = FastAPI()

# 處理請求驗證錯誤
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

# 處理 HTTP 異常
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    
    response = JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )
    
    # 添加標頭（如果有）
    if hasattr(exc, "headers") and exc.headers:
        for key, value in exc.headers.items():
            response.headers[key] = value
    
    return response

# 處理所有其他異常
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    content = {
        "status": "error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat(),
        "path": request.url.path
    }
    
    # 在開發環境中添加詳細錯誤信息
    if DEBUG:
        content.update({
            "detail": str(exc),
            "exception_type": exc.__class__.__name__,
            "traceback": traceback.format_exc().split("\n")
        })
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content
    )

# 示例路由
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id < 0:
        raise HTTPException(status_code=400, detail="Item ID must be positive")
    if item_id == 0:
        raise ValueError("Zero is not a valid item ID")
    return {"item_id": item_id, "name": f"Item {item_id}"}
```

## 小結

全局異常處理是構建健壯 FastAPI 應用的關鍵部分：

- **處理內建異常**：為 `RequestValidationError`、`HTTPException` 等內建異常提供自定義處理器
- **捕獲未處理的異常**：使用全局異常處理器捕獲所有未處理的異常，防止應用崩潰
- **提供一致的錯誤響應**：確保所有錯誤響應遵循一致的格式，便於客戶端處理
- **區分環境**：在開發環境中提供詳細的錯誤信息，在生產環境中隱藏敏感信息
- **整合日誌記錄**：記錄所有錯誤，便於調試和監控
- **異常分類**：根據異常類型提供不同的錯誤響應
- **整合錯誤報告服務**：將錯誤發送到 Sentry 等錯誤報告服務

通過實施全面的全局異常處理策略，您可以創建更健壯、更易於維護的 FastAPI 應用，提供更好的開發者和用戶體驗。
