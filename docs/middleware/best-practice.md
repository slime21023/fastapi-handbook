# FastAPI 中間件最佳實踐

本章節將介紹 FastAPI 中間件開發和使用的最佳實踐，幫助開發者構建更加健壯、高效的應用。

## 中間件設計原則

### 單一職責原則

每個中間件應該只負責一個明確的功能，避免混合多種不相關的邏輯：

```python
# 不推薦：混合多種功能的中間件
@app.middleware("http")
async def mixed_middleware(request: Request, call_next):
    # 記錄請求 + 身份驗證 + 請求限流 + 修改響應...
    # 太多職責在一個中間件中!
    
# 推薦：分離關注點的中間件
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    print(f"Request: {request.method} {request.url.path}")
    return await call_next(request)

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # 只處理身份驗證
    pass

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # 只處理速率限制
    pass
```

### 可配置性

設計中間件時，應該考慮其可配置性，使其能夠適應不同的使用場景：

```python
class ConfigurableMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app,
        include_paths: List[str] = None,
        exclude_paths: List[str] = None,
        include_methods: List[str] = None,
        debug_mode: bool = False
    ):
        super().__init__(app)
        # 將所有配置保存為實例變量
        self.include_paths = self._compile_patterns(include_paths)
        self.exclude_paths = self._compile_patterns(exclude_paths)
        self.include_methods = [m.upper() for m in (include_methods or [])]
        self.debug_mode = debug_mode
    
    # 其餘實現...
```

### 可測試性

將核心邏輯與框架集成分離，便於單元測試：

```python
# 將核心邏輯與中間件框架分離
class RateLimiter:
    def __init__(self, limit: int = 100, window: int = 60):
        self.limit = limit
        self.window = window
        self.requests = {}
    
    def is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        # 核心限流邏輯
        pass

# 使用核心邏輯的中間件
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: RateLimiter = None, **kwargs):
        super().__init__(app)
        self.limiter = limiter or RateLimiter(**kwargs)
    
    async def dispatch(self, request: Request, call_next):
        # 使用獨立的限流器
        pass
```

## 性能優化

### 避免阻塞操作

在中間件中應避免任何阻塞操作，以免影響整個應用的性能：

```python
# 不推薦：在中間件中使用阻塞操作
@app.middleware("http")
async def blocking_middleware(request: Request, call_next):
    time.sleep(1)  # 阻塞整個事件循環！
    
    # 阻塞 I/O
    with open("log.txt", "a") as f:
        f.write(f"{request.method} {request.url.path}\n")
    
    return await call_next(request)

# 推薦：使用非阻塞操作
@app.middleware("http")
async def non_blocking_middleware(request: Request, call_next):
    await asyncio.sleep(1)  # 非阻塞
    
    # 非阻塞 I/O
    async with aiofiles.open("log.txt", "a") as f:
        await f.write(f"{request.method} {request.url.path}\n")
    
    return await call_next(request)
```

### 緩存重複計算

對於重複計算的結果，應該使用緩存來提高性能：

```python
class CachingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.get_config = lru_cache(maxsize=100)(self._get_config)
    
    async def dispatch(self, request: Request, call_next):
        # 獲取配置（結果會被緩存）
        config = self.get_config(request.url.path)
        request.state.config = config
        return await call_next(request)
    
    def _get_config(self, path: str) -> dict:
        """昂貴的操作，但結果會被緩存"""
        # 模擬昂貴的計算或數據庫查詢
        time.sleep(0.1)
        # 返回路徑相關的配置
        return {"rate_limit": 100, "cache_ttl": 300}
```

### 延遲加載

對於不是每個請求都需要的資源，應該採用延遲加載策略：

```python
class ExternalServiceMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._session = None  # 不立即創建會話
    
    @property
    async def session(self):
        """延遲初始化 HTTP 會話"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/external/"):
            # 只有在需要時才獲取會話
            session = await self.session
            # 使用會話...
        
        return await call_next(request)
```

## 錯誤處理與穩健性

### 全面的錯誤處理

中間件應該處理所有可能發生的異常，確保應用的穩定性：

```python
class RobustMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            # 中間件邏輯...
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # 處理 FastAPI 的 HTTP 異常
            logger.warning(f"HTTP Exception: {e.detail}")
            raise  # 重新拋出 HTTP 異常
            
        except Exception as e:
            # 處理未捕獲的異常
            logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
            
            # 返回用戶友好的錯誤信息
            return JSONResponse(
                status_code=500,
                content={"detail": "An unexpected error occurred"}
            )
```

### 優雅降級

當某些功能不可用時，中間件應該能夠優雅降級：

```python
class ExternalServiceMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, service_url: str, timeout: float = 1.0):
        super().__init__(app)
        self.service_url = service_url
        self.timeout = timeout
        self.fallback_data = {"status": "offline", "data": []}  # 後備數據
    
    async def dispatch(self, request: Request, call_next):
        try:
            # 嘗試獲取外部服務數據，設置超時
            async with asyncio.timeout(self.timeout):
                # 調用外部服務...
                request.state.service_data = data
        except Exception as e:
            # 服務不可用，使用後備數據
            logger.warning(f"External service unavailable: {str(e)}")
            request.state.service_data = self.fallback_data
        
        # 繼續處理請求
        return await call_next(request)
```

### 重試機制

對於不可靠的操作，可以實現重試機制：

```python
class RetryMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_retries: int = 3, retry_delay: float = 0.1):
        super().__init__(app)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def retry_with_backoff(self, func, *args, **kwargs):
        """使用退避策略重試函數"""
        retries = 0
        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    raise
                
                # 計算延遲時間（指數退避）
                delay = self.retry_delay * (2 ** (retries - 1))
                await asyncio.sleep(delay)
    
    async def dispatch(self, request: Request, call_next):
        # 對於關鍵 API，使用重試機制
        if request.url.path.startswith("/api/critical/"):
            return await self.retry_with_backoff(call_next, request)
        else:
            return await call_next(request)
```

## 中間件組織與管理

### 中間件分組

對相關的中間件進行分組，便於管理和配置：

```python
class MiddlewareGroup:
    def __init__(self, app: FastAPI, name: str):
        self.app = app
        self.name = name
        self.middleware_list = []
    
    def add(self, middleware_class, **options):
        """添加中間件到組"""
        self.middleware_list.append((middleware_class, options))
        return self
    
    def apply(self):
        """將所有中間件應用到應用"""
        # 反向應用，確保執行順序與添加順序一致
        for middleware_class, options in reversed(self.middleware_list):
            self.app.add_middleware(middleware_class, **options)
        return self.app

# 使用示例
security_middleware = MiddlewareGroup(app, "security")
security_middleware.add(
    CORSMiddleware,
    allow_origins=["https://example.com"]
).add(
    AuthMiddleware,
    secret_key="your-secret-key"
)

# 應用安全中間件組
security_middleware.apply()
```

### 條件中間件應用

根據環境或配置條件應用不同的中間件：

```python
# 獲取環境配置
ENV = os.environ.get("APP_ENV", "development")
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# 根據環境應用不同的中間件
if ENV == "development":
    # 開發環境中間件
    app.add_middleware(
        LoggingMiddleware,
        log_level="debug",
        include_headers=True
    )
elif ENV == "production":
    # 生產環境中間件
    app.add_middleware(
        LoggingMiddleware,
        log_level="info"
    )
    
    app.add_middleware(
        RateLimitMiddleware,
        limit=100
    )

# 所有環境通用的中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENV == "development" else ["https://example.com"]
)
```

## 監控與可觀測性

### 中間件性能指標

收集和報告中間件性能指標，幫助識別潛在問題：

```python
class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.request_times = {}  # 儲存請求處理時間
    
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        endpoint = f"{method} {path}"
        
        # 記錄開始時間
        start_time = time.time()
        
        # 處理請求
        response = await call_next(request)
        
        # 計算處理時間
        process_time = time.time() - start_time
        
        # 更新指標
        if endpoint not in self.request_times:
            self.request_times[endpoint] = []
        
        self.request_times[endpoint].append(process_time)
        
        # 限制列表大小
        if len(self.request_times[endpoint]) > 100:
            self.request_times[endpoint] = self.request_times[endpoint][-100:]
        
        # 每 100 個請求報告一次
        if len(self.request_times[endpoint]) % 100 == 0:
            avg_time = sum(self.request_times[endpoint]) / len(self.request_times[endpoint])
            print(f"Average processing time for {endpoint}: {avg_time:.4f}s")
        
        return response
```

### 請求跟蹤

添加請求跟蹤標識符，便於日誌關聯和問題排查：

```python
class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 生成唯一的請求 ID
        request_id = str(uuid.uuid4())
        
        # 將請求 ID 添加到請求狀態
        request.state.request_id = request_id
        
        # 記錄請求開始
        print(f"[{request_id}] Request started: {request.method} {request.url.path}")
        
        # 處理請求
        response = await call_next(request)
        
        # 將請求 ID 添加到響應頭
        response.headers["X-Request-ID"] = request_id
        
        # 記錄請求結束
        print(f"[{request_id}] Request completed: Status {response.status_code}")
        
        return response
```

## 常見陷阱與解決方案

### 中間件執行順序

FastAPI 中間件的執行順序與添加順序相反，後添加的先執行：

```python
# 中間件執行順序：C -> B -> A
app.add_middleware(MiddlewareA)
app.add_middleware(MiddlewareB)
app.add_middleware(MiddlewareC)

# 如果需要確保特定順序，可以使用數字前綴命名
app.add_middleware(Middleware3_Last)
app.add_middleware(Middleware2_Middle)
app.add_middleware(Middleware1_First)
```

### 避免修改不可變對象

請求和響應的某些部分是不可變的，應該小心處理：

```python
# 錯誤：嘗試直接修改請求的 URL
@app.middleware("http")
async def wrong_middleware(request: Request, call_next):
    # 這會失敗，因為 URL 是不可變的
    request.url.path = "/modified" + request.url.path
    return await call_next(request)

# 正確：使用請求狀態存儲額外信息
@app.middleware("http")
async def correct_middleware(request: Request, call_next):
    # 將原始路徑存儲在請求狀態中
    request.state.original_path = request.url.path
    return await call_next(request)
```

### 正確處理響應流

對於流式響應，中間件需要特殊處理：

```python
class StreamMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # 檢查是否為流式響應
        if response.__class__.__name__ == "StreamingResponse":
            # 不要嘗試讀取或修改響應體
            return response
        
        # 處理普通響應
        # ...
        
        return response
```

## 小結

本章介紹了 FastAPI 中間件的最佳實踐，包括設計原則、性能優化、錯誤處理、中間件組織與管理以及監控等方面。遵循這些最佳實踐，可以幫助開發者構建更加健壯、高效和可維護的 FastAPI 應用。

關鍵要點：
- 遵循單一職責原則，每個中間件只處理一個關注點
- 設計可配置、可測試的中間件
- 避免阻塞操作，使用緩存和延遲加載提高性能
- 實現全面的錯誤處理和優雅降級機制
- 根據環境和需求組織和管理中間件
- 添加監控和可觀測性功能
- 注意中間件執行順序和不可變對象處理

通過合理應用這些最佳實踐，可以充分發揮 FastAPI 中間件的強大功能，構建高質量的 Web API。