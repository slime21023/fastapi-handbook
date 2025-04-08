# FastAPI 中間件應用場景 

本章節將探討 FastAPI 中間件的常見應用場景，並提供實用的實例代碼，幫助開發者理解如何在實際項目中有效利用中間件。

## 請求日誌記錄中間件

日誌記錄是中間件最常見的應用場景之一。通過中間件，我們可以統一記錄所有請求的詳細信息，包括請求方法、路徑、處理時間、狀態碼等。

```python
import time
import logging
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("app")

app = FastAPI()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 記錄請求開始時間
        start_time = time.time()
        
        # 收集請求信息
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        client_host = request.client.host if request.client else "unknown"
        
        # 記錄請求開始
        logger.info(f"Request started: {method} {path} from {client_host} with params {query_params}")
        
        try:
            # 處理請求
            response = await call_next(request)
            
            # 計算處理時間
            process_time = time.time() - start_time
            
            # 記錄成功的請求
            logger.info(
                f"Request completed: {method} {path} - Status: {response.status_code} - "
                f"Duration: {process_time:.4f}s"
            )
            
            return response
            
        except Exception as e:
            # 記錄失敗的請求
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {method} {path} - Error: {str(e)} - "
                f"Duration: {process_time:.4f}s"
            )
            raise

app.add_middleware(LoggingMiddleware)
```

## 身份驗證與授權中間件

中間件是實現身份驗證和授權邏輯的理想位置，可以在請求到達路由處理函數之前驗證用戶身份。

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from typing import List, Optional

app = FastAPI()

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app, 
        secret_key: str,
        exclude_paths: List[str] = None,
        algorithm: str = "HS256"
    ):
        super().__init__(app)
        self.secret_key = secret_key
        self.exclude_paths = exclude_paths or ["/login", "/docs", "/openapi.json"]
        self.algorithm = algorithm
    
    async def dispatch(self, request: Request, call_next):
        # 檢查是否為排除路徑
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # 獲取授權頭
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid authentication token"}
            )
        
        token = auth_header.split(" ")[1]
        
        try:
            # 驗證 JWT 令牌
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # 將用戶信息添加到請求狀態
            request.state.user = payload
            request.state.user_id = payload.get("sub")
            request.state.user_role = payload.get("role")
            
            # 繼續處理請求
            return await call_next(request)
            
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token has expired"}
            )
        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authentication token"}
            )

# 註冊中間件
app.add_middleware(
    AuthMiddleware,
    secret_key="your-secret-key",
    exclude_paths=["/login", "/register", "/docs", "/openapi.json"]
)

# 在路由中使用請求狀態中的用戶信息
@app.get("/profile")
async def get_profile(request: Request):
    user = request.state.user
    return {"user_id": user.get("sub"), "username": user.get("username")}
```

## CORS 處理中間件

跨域資源共享 (CORS) 是 Web 應用中常見的需求，FastAPI 提供了內置的 CORS 中間件：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    # 允許的源列表
    allow_origins=[
        "http://localhost:3000",
        "https://frontend.example.com"
    ],
    # 是否允許發送憑證（如 cookies）
    allow_credentials=True,
    # 允許的 HTTP 方法
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    # 允許的 HTTP 頭
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    # 允許瀏覽器緩存預檢請求的時間（秒）
    max_age=600,
)
```

## 請求限流中間件

為了防止 API 被濫用，可以實現請求限流中間件，限制特定時間內的請求次數：

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict

app = FastAPI()

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app, 
        limit: int = 10,
        window: int = 60,
        exclude_paths: list = None
    ):
        super().__init__(app)
        self.limit = limit  # 每個窗口允許的最大請求數
        self.window = window  # 窗口大小（秒）
        self.exclude_paths = exclude_paths or []
        self.requests = defaultdict(list)  # 儲存每個 IP 的請求時間
    
    async def dispatch(self, request: Request, call_next):
        # 檢查是否為排除路徑
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # 獲取客戶端 IP
        client_ip = request.client.host if request.client else "unknown"
        
        # 當前時間
        current_time = time.time()
        
        # 清理過期的請求記錄
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < self.window
        ]
        
        # 檢查是否超過限制
        if len(self.requests[client_ip]) >= self.limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests",
                    "retry_after": self.window - (current_time - self.requests[client_ip][0])
                }
            )
        
        # 記錄當前請求
        self.requests[client_ip].append(current_time)
        
        # 處理請求
        return await call_next(request)

# 註冊中間件
app.add_middleware(
    RateLimitMiddleware,
    limit=100,  # 每分鐘 100 個請求
    window=60,  # 1 分鐘窗口
    exclude_paths=["/docs", "/openapi.json"]
)
```

## 響應壓縮中間件

對於大型響應，可以使用壓縮中間件減少傳輸數據量，提高性能：

```python
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

# 註冊 GZip 壓縮中間件
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000  # 僅壓縮大於 1000 字節的響應
)
```

## 全局異常處理中間件

中間件可以捕獲應用中的所有異常，提供統一的錯誤處理機制：

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
import logging

logger = logging.getLogger("app")

app = FastAPI()

class ExceptionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException as e:
            # 處理 FastAPI 的 HTTPException
            logger.warning(f"HTTP Exception: {e.detail} (status_code={e.status_code})")
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        except Exception as e:
            # 處理未捕獲的異常
            error_id = str(uuid.uuid4())
            
            # 記錄詳細錯誤信息
            logger.error(
                f"Unhandled exception: {str(e)} (error_id={error_id})\n"
                f"{traceback.format_exc()}"
            )
            
            # 返回用戶友好的錯誤信息
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "An unexpected error occurred",
                    "error_id": error_id
                }
            )

app.add_middleware(ExceptionMiddleware)
```

## 性能監控中間件

監控 API 端點的性能，識別可能的瓶頸：

```python
import time
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
import statistics

app = FastAPI()

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, threshold_ms: float = 500):
        super().__init__(app)
        self.threshold_ms = threshold_ms
        self.request_times = {}  # 儲存每個路徑的處理時間
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        # 計算處理時間（毫秒）
        process_time = (time.time() - start_time) * 1000
        
        # 獲取請求路徑
        path = request.url.path
        
        # 更新路徑的處理時間統計
        if path not in self.request_times:
            self.request_times[path] = []
        
        self.request_times[path].append(process_time)
        
        # 保留最近 100 個請求的數據
        if len(self.request_times[path]) > 100:
            self.request_times[path].pop(0)
        
        # 如果處理時間超過閾值，記錄警告
        if process_time > self.threshold_ms:
            print(f"WARNING: Slow request detected - {request.method} {path} took {process_time:.2f}ms")
        
        # 每 100 個請求計算一次統計數據
        if len(self.request_times[path]) % 100 == 0:
            times = self.request_times[path]
            avg_time = statistics.mean(times)
            p95_time = sorted(times)[int(len(times) * 0.95)]
            
            print(f"Performance stats for {path}:")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  95th percentile: {p95_time:.2f}ms")
            print(f"  Min: {min(times):.2f}ms")
            print(f"  Max: {max(times):.2f}ms")
        
        # 將處理時間添加到響應頭
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        return response

app.add_middleware(PerformanceMonitoringMiddleware, threshold_ms=200)
```

## 國際化 (i18n) 中間件

根據請求頭或 URL 參數自動設置語言：

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, List

app = FastAPI()

class I18nMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app, 
        default_language: str = "en",
        supported_languages: List[str] = None
    ):
        super().__init__(app)
        self.default_language = default_language
        self.supported_languages = supported_languages or ["en"]
    
    async def dispatch(self, request: Request, call_next):
        # 嘗試從查詢參數獲取語言
        lang_param = request.query_params.get("lang")
        
        # 嘗試從 Accept-Language 頭獲取語言
        accept_language = request.headers.get("Accept-Language", "")
        accept_languages = [
            lang.split(";")[0].strip() 
            for lang in accept_language.split(",")
        ]
        
        # 確定要使用的語言
        language = None
        
        # 優先使用查詢參數中的語言
        if lang_param and lang_param in self.supported_languages:
            language = lang_param
        else:
            # 否則嘗試使用 Accept-Language 頭
            for lang in accept_languages:
                if lang in self.supported_languages:
                    language = lang
                    break
        
        # 如果沒有找到支持的語言，使用默認語言
        if not language:
            language = self.default_language
        
        # 將語言設置到請求狀態
        request.state.language = language
        
        # 處理請求
        response = await call_next(request)
        
        return response

# 註冊中間件
app.add_middleware(
    I18nMiddleware,
    default_language="en",
    supported_languages=["en", "zh", "ja", "ko", "fr", "es"]
)

# 在路由中使用語言設置
@app.get("/greeting")
async def greeting(request: Request):
    language = request.state.language
    
    greetings = {
        "en": "Hello, world!",
        "zh": "你好，世界！",
        "ja": "こんにちは、世界！",
        "ko": "안녕하세요, 세계!",
        "fr": "Bonjour, monde!",
        "es": "¡Hola, mundo!"
    }
    
    return {"message": greetings.get(language, greetings["en"])}
```

## 小結

本章介紹了 FastAPI 中間件的多種實用場景，包括日誌記錄、身份驗證、CORS 處理、請求限流、響應壓縮、異常處理、性能監控和國際化等。這些實例展示了中間件在解決橫切關注點方面的強大能力，可以幫助開發者構建更加健壯、安全和高效的 API。

通過合理組合和配置這些中間件，開發者可以為 FastAPI 應用添加各種功能，而無需修改核心業務邏輯。這種關注點分離的設計模式，使得代碼更加模塊化、可維護和可擴展。

