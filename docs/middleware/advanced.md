# FastAPI 中間件進階技巧 

本章節將深入探討 FastAPI 中間件的進階技巧，包括異步中間件優化、中間件與依賴注入的結合、中間件上下文管理、中間件測試策略以及其他高級應用場景。

## 異步中間件優化

FastAPI 建立在 ASGI 標準之上，充分利用 Python 的異步特性可以顯著提高應用性能。以下是一些異步中間件的優化技巧：

### 避免阻塞操作

在異步中間件中，應避免使用阻塞操作，如同步 I/O、長時間計算等：

```python
from fastapi import FastAPI, Request
import aiohttp
import asyncio

app = FastAPI()

@app.middleware("http")
async def external_service_middleware(request: Request, call_next):
    # 錯誤示例：在異步中間件中使用阻塞操作
    # import requests
    # response = requests.get("https://api.example.com/data")  # 阻塞!
    
    # 正確示例：使用異步 HTTP 客戶端
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com/data") as response:
            data = await response.json()
    
    # 將數據添加到請求狀態
    request.state.external_data = data
    
    # 繼續處理請求
    return await call_next(request)
```

### 使用異步上下文管理器

利用異步上下文管理器可以更優雅地處理資源獲取和釋放：

```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

@asynccontextmanager
async def get_db_connection():
    # 假設這是一個異步數據庫連接
    conn = await create_async_connection()
    try:
        yield conn
    finally:
        await conn.close()

class DatabaseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        async with get_db_connection() as conn:
            # 將數據庫連接添加到請求狀態
            request.state.db = conn
            
            # 處理請求
            response = await call_next(request)
            
            # 連接會在上下文管理器退出時自動關閉
            return response

app = FastAPI()
app.add_middleware(DatabaseMiddleware)
```

### 並行處理

在中間件中可以使用 `asyncio.gather` 並行執行多個異步任務：

```python
@app.middleware("http")
async def parallel_tasks_middleware(request: Request, call_next):
    # 定義需要並行執行的任務
    async def fetch_user_data():
        # 假設這是從數據庫獲取用戶數據的異步操作
        await asyncio.sleep(0.1)  # 模擬 I/O 延遲
        return {"user_id": 123, "name": "John Doe"}
    
    async def fetch_metrics():
        # 假設這是從指標系統獲取數據的異步操作
        await asyncio.sleep(0.1)  # 模擬 I/O 延遲
        return {"api_calls": 1000, "error_rate": 0.01}
    
    # 並行執行任務
    user_data, metrics = await asyncio.gather(
        fetch_user_data(),
        fetch_metrics()
    )
    
    # 將結果存儲到請求狀態
    request.state.user_data = user_data
    request.state.metrics = metrics
    
    # 繼續處理請求
    return await call_next(request)
```

## 中間件與依賴注入結合

FastAPI 的中間件和依賴注入系統可以協同工作，實現更強大的功能：

### 在中間件中預處理依賴項

```python
from fastapi import FastAPI, Request, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Optional

# 定義一個依賴項
async def get_current_user(request: Request) -> Optional[Dict]:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    # 這裡省略了實際的令牌驗證邏輯
    return {"user_id": 123, "username": "john_doe"}

class AuthPreprocessingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 預先解析用戶信息
        user = await get_current_user(request)
        
        # 將用戶信息存儲到請求狀態
        request.state.user = user
        
        # 處理請求
        response = await call_next(request)
        return response

app = FastAPI()
app.add_middleware(AuthPreprocessingMiddleware)

# 在路由中使用預處理的用戶信息
@app.get("/user/profile")
async def get_profile(request: Request):
    user = request.state.user
    if not user:
        return {"detail": "Not authenticated"}
    return {"profile": user}
```

### 自定義中間件依賴項

創建可在多個中間件之間共享的依賴項：

```python
from fastapi import FastAPI, Request, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, TypeVar, Generic, Optional

T = TypeVar('T')

class MiddlewareDependency(Generic[T]):
    def __init__(self, dependency: Callable[..., T]):
        self.dependency = dependency
    
    async def resolve(self, request: Request) -> T:
        return await self.dependency(request)

# 創建一個配置依賴項
async def get_app_config(request: Request):
    # 這裡可以是從數據庫或配置文件加載配置
    return {
        "feature_flags": {
            "new_ui": True,
            "beta_features": False
        },
        "limits": {
            "max_requests_per_minute": 100
        }
    }

config_dependency = MiddlewareDependency(get_app_config)

class ConfigMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 解析配置依賴項
        config = await config_dependency.resolve(request)
        
        # 將配置存儲到請求狀態
        request.state.config = config
        
        # 處理請求
        response = await call_next(request)
        return response

app = FastAPI()
app.add_middleware(ConfigMiddleware)

# 在路由中使用配置
@app.get("/features")
async def get_features(request: Request):
    config = request.state.config
    return {"features": config["feature_flags"]}
```

## 中間件上下文管理

在複雜應用中，有效管理中間件上下文可以簡化代碼並提高可維護性：

### 請求上下文管理

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from contextvars import ContextVar
from typing import Optional, Dict, Any

# 定義上下文變量
request_id_var: ContextVar[str] = ContextVar("request_id", default=None)
user_var: ContextVar[Optional[Dict]] = ContextVar("user", default=None)

# 上下文管理器
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 生成請求 ID
        request_id = str(uuid.uuid4())
        
        # 設置上下文變量
        request_id_token = request_id_var.set(request_id)
        
        # 獲取用戶信息（如果有）
        auth_header = request.headers.get("Authorization")
        user = None
        if auth_header and auth_header.startswith("Bearer "):
            # 這裡省略了實際的令牌驗證邏輯
            user = {"user_id": 123, "username": "john_doe"}
        
        user_token = user_var.set(user)
        
        try:
            # 處理請求
            response = await call_next(request)
            
            # 添加請求 ID 到響應頭
            response.headers["X-Request-ID"] = request_id
            
            return response
        finally:
            # 重置上下文變量
            request_id_var.reset(request_id_token)
            user_var.reset(user_token)

app = FastAPI()
app.add_middleware(RequestContextMiddleware)

# 在任何地方獲取當前請求 ID
def get_current_request_id() -> str:
    return request_id_var.get()

# 在任何地方獲取當前用戶
def get_current_user() -> Optional[Dict]:
    return user_var.get()

# 使用上下文變量
@app.get("/context-demo")
async def context_demo():
    request_id = get_current_request_id()
    user = get_current_user()
    
    return {
        "request_id": request_id,
        "user": user
    }
```

### 分層中間件上下文

處理複雜的多層中間件場景：

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any, List
import time

app = FastAPI()

class ContextualizedMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, name: str):
        super().__init__(app)
        self.name = name
    
    async def dispatch(self, request: Request, call_next):
        # 初始化上下文堆疊（如果不存在）
        if not hasattr(request.state, "middleware_context"):
            request.state.middleware_context = []
            request.state.timing_data = {}
        
        # 記錄進入中間件的時間
        start_time = time.time()
        
        # 將當前中間件添加到上下文堆疊
        context = {"name": self.name, "entered_at": start_time}
        request.state.middleware_context.append(context)
        
        try:
            # 處理請求
            response = await call_next(request)
            
            # 記錄退出中間件的時間
            end_time = time.time()
            elapsed = end_time - start_time
            
            # 更新計時數據
            request.state.timing_data[self.name] = elapsed
            
            return response
        finally:
            # 從上下文堆疊中移除當前中間件
            request.state.middleware_context.pop()

# 註冊多個中間件實例
app.add_middleware(ContextualizedMiddleware, name="outer")
app.add_middleware(ContextualizedMiddleware, name="middle")
app.add_middleware(ContextualizedMiddleware, name="inner")

@app.get("/context-stack")
async def get_context_stack(request: Request):
    return {
        "current_context": request.state.middleware_context,
        "timing_data": request.state.timing_data
    }
```

## 條件中間件執行

根據請求特性有條件地執行中間件邏輯：

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
import re

app = FastAPI()

class ConditionalMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app, 
        func,
        include_paths: List[str] = None,
        exclude_paths: List[str] = None,
        include_methods: List[str] = None
    ):
        super().__init__(app)
        self.func = func
        self.include_paths = [re.compile(path) for path in (include_paths or [])]
        self.exclude_paths = [re.compile(path) for path in (exclude_paths or [])]
        self.include_methods = [m.upper() for m in (include_methods or [])]
    
    def should_process(self, request: Request) -> bool:
        path = request.url.path
        method = request.method
        
        # 檢查排除路徑
        if any(pattern.match(path) for pattern in self.exclude_paths):
            return False
        
        # 檢查包含路徑
        if self.include_paths and not any(pattern.match(path) for pattern in self.include_paths):
            return False
        
        # 檢查包含方法
        if self.include_methods and method not in self.include_methods:
            return False
        
        return True
    
    async def dispatch(self, request: Request, call_next):
        if self.should_process(request):
            # 執行中間件邏輯
            return await self.func(request, call_next)
        else:
            # 跳過中間件邏輯
            return await call_next(request)

# 定義中間件邏輯
async def rate_limit_logic(request: Request, call_next):
    # 這裡實現限流邏輯
    print(f"Rate limiting applied to {request.url.path}")
    return await call_next(request)

# 註冊條件中間件
app.add_middleware(
    ConditionalMiddleware,
    func=rate_limit_logic,
    include_paths=[r"/api/.*"],  # 僅對 /api/ 開頭的路徑應用
    exclude_paths=[r"/api/public/.*"],  # 排除 /api/public/ 開頭的路徑
    include_methods=["POST", "PUT", "DELETE"]  # 僅對寫操作應用
)
```

## 中間件工廠模式

使用工廠模式創建可配置的中間件：

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Dict, Any, Optional
import json
import time

def create_logging_middleware(
    log_format: str = "default",
    include_headers: bool = False,
    include_body: bool = False,
    log_level: str = "info"
):
    """中間件工廠函數，創建可配置的日誌中間件"""
    
    class LoggingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # 記錄請求開始時間
            start_time = time.time()
            
            # 收集請求信息
            request_info = {
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_host": request.client.host if request.client else "unknown",
            }
            
            # 根據配置添加頭信息
            if include_headers:
                request_info["headers"] = dict(request.headers)
            
            # 根據配置添加請求體
            if include_body and request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        try:
                            # 嘗試解析 JSON
                            request_info["body"] = json.loads(body)
                        except json.JSONDecodeError:
                            # 如果不是 JSON，則保存原始字符串
                            request_info["body"] = body.decode("utf-8", errors="replace")
                except Exception:
                    pass
            
            # 記錄請求信息
            log_message = f"Request: {json.dumps(request_info)}"
            if log_level == "debug":
                print(f"DEBUG: {log_message}")
            else:
                print(f"INFO: {log_message}")
            
            # 處理請求
            response = await call_next(request)
            
            # 計算處理時間
            process_time = time.time() - start_time
            
            # 記錄響應信息
            response_info = {
                "status_code": response.status_code,
                "process_time": f"{process_time:.4f}s"
            }
            
            # 根據配置添加響應頭
            if include_headers:
                response_info["headers"] = dict(response.headers)
            
            # 記錄響應信息
            log_message = f"Response: {json.dumps(response_info)}"
            if log_level == "debug":
                print(f"DEBUG: {log_message}")
            else:
                print(f"INFO: {log_message}")
            
            return response
    
    return LoggingMiddleware

app = FastAPI()

# 使用工廠函數創建不同配置的中間件
app.add_middleware(create_logging_middleware(log_level="debug", include_headers=True))

# 可以為不同的應用創建不同配置的中間件
api_app = FastAPI()
api_app.add_middleware(create_logging_middleware(include_body=True, include_headers=True))

# 掛載子應用
app.mount("/api", api_app)
```

## 中間件測試策略

有效測試中間件是確保其正確性和穩定性的關鍵：

```python
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

# 待測試的中間件
class HeaderMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str, header_value: str):
        super().__init__(app)
        self.header_name = header_name
        self.header_value = header_value
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers[self.header_name] = self.header_value
        return response

# 測試函數
def test_header_middleware():
    app = FastAPI()
    app.add_middleware(HeaderMiddleware, header_name="X-Test-Header", header_value="test-value")
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    client = TestClient(app)
    response = client.get("/test")
    
    assert response.status_code == 200
    assert response.headers["X-Test-Header"] == "test-value"
    assert response.json() == {"message": "test"}

# 測試中間件執行順序
def test_middleware_order():
    app = FastAPI()
    execution_order = []
    
    class OrderMiddleware(BaseHTTPMiddleware):
        def __init__(self, app, name: str):
            super().__init__(app)
            self.name = name
        
        async def dispatch(self, request: Request, call_next):
            execution_order.append(f"{self.name}_before")
            response = await call_next(request)
            execution_order.append(f"{self.name}_after")
            return response
    
    app.add_middleware(OrderMiddleware, name="outer")
    app.add_middleware(OrderMiddleware, name="inner")
    
    @app.get("/test")
    async def test_endpoint():
        execution_order.append("endpoint")
        return {"message": "test"}
    
    client = TestClient(app)
    client.get("/test")
    
    # 驗證執行順序
    assert execution_order == [
        "outer_before",
        "inner_before",
        "endpoint",
        "inner_after",
        "outer_after"
    ]
```

## 中間件與 WebSocket 支持

FastAPI 中間件不僅可以處理 HTTP 請求，還可以處理 WebSocket 連接：

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.middleware.base import BaseHTTPMiddleware
import json

app = FastAPI()

class WebSocketMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if "websocket" in request.scope["type"]:
            # 處理 WebSocket 連接
            websocket = WebSocket(request.scope, request.receive, request.send)
            
            # 執行自定義 WebSocket 處理邏輯
            try:
                await websocket.accept()
                
                # 發送歡迎消息
                await websocket.send_text("Welcome to the WebSocket server!")
                
                # 處理消息
                while True:
                    data = await websocket.receive_text()
                    await websocket.send_text(f"Echo: {data}")
            except WebSocketDisconnect:
                print("WebSocket disconnected")
            
            return None  # WebSocket 連接已處理
        else:
            # 處理普通 HTTP 請求
            return await call_next(request)

# 註冊中間件
app.add_middleware(WebSocketMiddleware)

# 定義 WebSocket 端點
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 這個函數不會被調用，因為中間件已經處理了 WebSocket 連接
    pass
```

## 動態中間件註冊與管理

在運行時動態管理中間件：

```python
from fastapi import FastAPI, Request, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, List, Type, Optional
import inspect

app = FastAPI()

# 中間件註冊表
middleware_registry: Dict[str, Type[BaseHTTPMiddleware]] = {}

# 動態中間件管理器
class DynamicMiddlewareManager:
    def __init__(self, app: FastAPI):
        self.app = app
        self.active_middleware: Dict[str, BaseHTTPMiddleware] = {}
    
    def register_middleware_class(self, name: str, middleware_class: Type[BaseHTTPMiddleware]):
        """註冊中間件類到註冊表"""
        middleware_registry[name] = middleware_class
    
    def activate_middleware(self, name: str, **kwargs):
        """激活並配置中間件"""
        if name not in middleware_registry:
            raise ValueError(f"Middleware '{name}' not found in registry")
        
        if name in self.active_middleware:
            raise ValueError(f"Middleware '{name}' is already active")
        
        # 創建中間件實例
        middleware_class = middleware_registry[name]
        
        # 檢查參數是否匹配
        init_params = inspect.signature(middleware_class.__init__).parameters
        valid_params = {k: v for k, v in kwargs.items() if k in init_params}
        
        # 實例化中間件
        middleware_instance = middleware_class(self.app, **valid_params)
        
        # 添加到應用
        self.app.add_middleware(type(middleware_instance), **valid_params)
        
        # 記錄激活的中間件
        self.active_middleware[name] = middleware_instance
        
        return f"Middleware '{name}' activated with parameters: {valid_params}"
    
    def get_active_middleware(self) -> List[str]:
        """獲取所有激活的中間件"""
        return list(self.active_middleware.keys())

# 創建中間件管理器
middleware_manager = DynamicMiddlewareManager(app)

# 定義一些示例中間件
class HeaderInjectionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str, header_value: str):
        super().__init__(app)
        self.header_name = header_name
        self.header_value = header_value
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers[self.header_name] = self.header_value
        return response

class RequestLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 100):
        super().__init__(app)
        self.max_requests = max_requests
        self.request_count = 0
    
    async def dispatch(self, request: Request, call_next):
        self.request_count += 1
        if self.request_count > self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"}
            )
        return await call_next(request)

# 註冊中間件類
middleware_manager.register_middleware_class("header_injection", HeaderInjectionMiddleware)
middleware_manager.register_middleware_class("request_limit", RequestLimitMiddleware)

# API 端點來管理中間件
@app.post("/admin/middleware/{name}/activate")
async def activate_middleware(name: str, params: Dict[str, Any]):
    return middleware_manager.activate_middleware(name, **params)

@app.get("/admin/middleware/active")
async def get_active_middleware():
    return {"active_middleware": middleware_manager.get_active_middleware()}
```

## 小結

本章深入探討了 FastAPI 中間件的進階技巧，包括異步中間件優化、中間件與依賴注入的結合、中間件上下文管理、條件中間件執行、中間件工廠模式、中間件測試策略、WebSocket 支持以及動態中間件註冊與管理。

這些進階技巧可以幫助開發者構建更加靈活、高效和可維護的 FastAPI 應用。通過合理組合和配置這些技巧，可以實現複雜的中間件架構，滿足各種應用需求。

中間件是 FastAPI 應用的重要組成部分，掌握這些進階技巧可以讓開發者更加得心應手地處理各種橫切關注點，提高代碼質量和應用性能。