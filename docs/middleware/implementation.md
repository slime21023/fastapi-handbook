# FastAPI 中間件實現

本章節將深入探討 FastAPI 中間件的實現方式，包括函數式中間件、類式中間件、註冊方法以及執行順序控制等關鍵內容。

## 函數式中間件實現

函數式中間件是 FastAPI 中最常用的中間件實現方式，通過裝飾器 `@app.middleware("http")` 來定義。

### 基本結構

```python
from fastapi import FastAPI, Request
from fastapi.responses import Response
from typing import Callable
import time

app = FastAPI()

@app.middleware("http")
async def timing_middleware(request: Request, call_next: Callable):
    start_time = time.time()
    
    # 處理請求
    response = await call_next(request)
    
    # 計算處理時間
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response
```

### 函數式中間件的參數

- **request**: `Request` 對象，包含當前 HTTP 請求的所有信息
- **call_next**: 一個可調用對象，用於調用下一個中間件或路由處理函數
- **返回值**: 必須返回一個 `Response` 對象

### 異步與同步

FastAPI 支持異步和同步中間件，但推薦使用異步方式以獲得更好的性能：

```python
# 異步中間件（推薦）
@app.middleware("http")
async def async_middleware(request: Request, call_next):
    response = await call_next(request)
    return response

# 同步中間件
@app.middleware("http")
def sync_middleware(request: Request, call_next):
    response = call_next(request)
    return response
```

## 類式中間件實現

除了函數式中間件，FastAPI 還支持基於類的中間件實現，這種方式更適合複雜的中間件邏輯。

### 基於 Starlette 的 BaseHTTPMiddleware

```python
from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

app = FastAPI()

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 處理請求
        response = await call_next(request)
        
        # 計算處理時間
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

# 註冊中間件
app.add_middleware(TimingMiddleware)
```

### 帶配置的類式中間件

類式中間件的一個優勢是可以輕鬆接受配置參數：

```python
class ConfigurableMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str = "X-Process-Time", log_time: bool = False):
        super().__init__(app)
        self.header_name = header_name
        self.log_time = log_time
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 添加自定義頭信息
        response.headers[self.header_name] = str(process_time)
        
        # 根據配置決定是否記錄時間
        if self.log_time:
            print(f"Request to {request.url.path} took {process_time:.4f} seconds")
            
        return response

# 註冊並配置中間件
app.add_middleware(
    ConfigurableMiddleware, 
    header_name="X-Custom-Process-Time", 
    log_time=True
)
```

## 中間件註冊方法

FastAPI 提供了兩種註冊中間件的方法：

### 1. 使用裝飾器（僅適用於函數式中間件）

```python
@app.middleware("http")
async def my_middleware(request, call_next):
    response = await call_next(request)
    return response
```

### 2. 使用 add_middleware 方法（適用於類式中間件）

```python
from starlette.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 內置中間件

FastAPI/Starlette 提供了幾個實用的內置中間件：

### CORS 中間件

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### GZip 壓縮中間件

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### 信任主機中間件

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["example.com", "*.example.com"]
)
```

## 中間件執行順序

中間件的執行順序遵循「洋蔥模型」，按照註冊順序執行前置處理，按照註冊的反序執行後置處理。

```python
@app.middleware("http")
async def middleware1(request, call_next):
    print("Middleware 1 - 前置處理")
    response = await call_next(request)
    print("Middleware 1 - 後置處理")
    return response

@app.middleware("http")
async def middleware2(request, call_next):
    print("Middleware 2 - 前置處理")
    response = await call_next(request)
    print("Middleware 2 - 後置處理")
    return response
```

執行順序將是：
```
Middleware 1 - 前置處理
Middleware 2 - 前置處理
路由處理函數執行
Middleware 2 - 後置處理
Middleware 1 - 後置處理
```

### 控制執行順序

使用 `add_middleware` 方法時，中間件的註冊順序決定了執行順序：

```python
# 先執行 CORSMiddleware，後執行 GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

## 請求和響應修改

中間件可以修改請求和響應的各個方面：

### 修改請求

```python
@app.middleware("http")
async def add_custom_header(request: Request, call_next):
    # 無法直接修改 request.headers，但可以通過其他方式傳遞信息
    request.state.custom_value = "some_value"
    response = await call_next(request)
    return response
```

### 修改響應

```python
@app.middleware("http")
async def modify_response(request: Request, call_next):
    response = await call_next(request)
    
    # 添加頭信息
    response.headers["X-Custom-Header"] = "custom_value"
    
    # 修改狀態碼
    if some_condition:
        response.status_code = 201
    
    return response
```

### 完全替換響應

```python
from fastapi.responses import JSONResponse

@app.middleware("http")
async def replace_response(request: Request, call_next):
    # 某些條件下，不調用下一個中間件，直接返回響應
    if request.url.path == "/blocked":
        return JSONResponse(
            status_code=403,
            content={"message": "Access denied"}
        )
    
    response = await call_next(request)
    return response
```

## 中間件中的錯誤處理

中間件可以捕獲和處理路由處理函數或其他中間件中的異常：

```python
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # 記錄錯誤
        print(f"Error: {str(e)}")
        
        # 返回自定義錯誤響應
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )
```

## 小結

本章詳細介紹了 FastAPI 中間件的實現方式，包括函數式和類式中間件的定義、註冊方法、執行順序控制以及請求和響應的修改。通過選擇合適的中間件實現方式，開發者可以更靈活地處理各種橫切關注點，提高代碼的可維護性和可擴展性。

