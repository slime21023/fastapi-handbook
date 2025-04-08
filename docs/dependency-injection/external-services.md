# FastAPI 依賴注入實戰：外部服務整合

## 簡介

現代 Web 應用很少是孤立的系統，它們通常需要與各種外部服務進行交互，如支付處理器、電子郵件服務、推送通知系統、第三方 API 等。FastAPI 的依賴注入系統為這些外部服務的整合提供了優雅且可維護的解決方案。

本文將探討如何利用 FastAPI 的依賴注入機制高效地整合外部服務，包括客戶端管理、錯誤處理、重試機制和測試策略。我們將通過實際的程式碼範例，展示不同的整合模式和最佳實踐。

## 外部服務整合的挑戰

整合外部服務時，我們通常面臨以下挑戰：

1. **客戶端生命週期管理**：何時創建和銷毀客戶端
2. **配置管理**：安全地存儲和使用 API 密鑰和其他敏感信息
3. **錯誤處理**：優雅地處理外部服務的故障
4. **性能優化**：避免不必要的連接和請求
5. **測試**：在不實際調用外部服務的情況下進行測試

FastAPI 的依賴注入系統可以幫助我們應對這些挑戰。

## 基本外部服務整合模式

### 1. 客戶端作為依賴項

最簡單的整合模式是將外部服務的客戶端作為依賴項提供。

```python
from fastapi import FastAPI, Depends
import httpx
from pydantic import BaseSettings

class Settings(BaseSettings):
    weather_api_key: str
    weather_api_url: str = "https://api.weatherapi.com/v1"
    
    class Config:
        env_file = ".env"

settings = Settings()
app = FastAPI()

# 客戶端依賴函數
def get_weather_client():
    return httpx.Client(
        base_url=settings.weather_api_url,
        params={"key": settings.weather_api_key}
    )

@app.get("/weather/{city}")
def get_weather(city: str, client: httpx.Client = Depends(get_weather_client)):
    response = client.get("/current.json", params={"q": city})
    response.raise_for_status()
    
    weather_data = response.json()
    return {
        "city": city,
        "temperature": weather_data["current"]["temp_c"],
        "condition": weather_data["current"]["condition"]["text"]
    }
```

**優點**：
- 簡單直接
- 易於理解和實現

**缺點**：
- 每個請求都創建新的客戶端
- 沒有連接池或重用機制

### 2. 共享客戶端實例

對於需要重用的客戶端，我們可以在應用啟動時創建一個共享實例。

```python
from fastapi import FastAPI, Depends
import httpx
from pydantic import BaseSettings

class Settings(BaseSettings):
    email_api_key: str
    email_api_url: str = "https://api.sendgrid.com/v3"
    
    class Config:
        env_file = ".env"

settings = Settings()
app = FastAPI()

# 全局客戶端
email_client = None

@app.on_event("startup")
async def startup_event():
    global email_client
    email_client = httpx.AsyncClient(
        base_url=settings.email_api_url,
        headers={"Authorization": f"Bearer {settings.email_api_key}"}
    )

@app.on_event("shutdown")
async def shutdown_event():
    global email_client
    if email_client:
        await email_client.aclose()

# 客戶端依賴函數
async def get_email_client():
    return email_client

@app.post("/send-email")
async def send_email(
    to: str, 
    subject: str, 
    content: str,
    client: httpx.AsyncClient = Depends(get_email_client)
):
    payload = {
        "personalizations": [{"to": [{"email": to}]}],
        "subject": subject,
        "content": [{"type": "text/plain", "value": content}],
        "from": {"email": "noreply@example.com"}
    }
    
    response = await client.post("/mail/send", json=payload)
    response.raise_for_status()
    
    return {"status": "Email sent successfully"}
```

**優點**：
- 客戶端在多個請求間共享
- 減少資源使用
- 支持連接池

**缺點**：
- 需要謹慎管理全局狀態
- 可能需要處理並發問題

## 進階整合模式

### 1. 服務類封裝

使用服務類封裝外部服務的交互邏輯，提供更好的抽象和可測試性。

```python
from fastapi import FastAPI, Depends, HTTPException
import httpx
from pydantic import BaseSettings, BaseModel
from typing import Optional

class Settings(BaseSettings):
    payment_api_key: str
    payment_api_url: str = "https://api.stripe.com/v1"
    
    class Config:
        env_file = ".env"

settings = Settings()
app = FastAPI()

# 請求和響應模型
class PaymentRequest(BaseModel):
    amount: int  # 金額（分）
    currency: str
    description: Optional[str] = None
    customer_id: Optional[str] = None

class PaymentResponse(BaseModel):
    payment_id: str
    status: str
    amount: int
    currency: str

# 支付服務類
class PaymentService:
    def __init__(self, client: httpx.Client):
        self.client = client
    
    def create_payment(self, payment: PaymentRequest) -> PaymentResponse:
        try:
            response = self.client.post(
                "/charges",
                data={
                    "amount": payment.amount,
                    "currency": payment.currency,
                    "description": payment.description,
                    "customer": payment.customer_id
                }
            )
            response.raise_for_status()
            
            payment_data = response.json()
            return PaymentResponse(
                payment_id=payment_data["id"],
                status=payment_data["status"],
                amount=payment_data["amount"],
                currency=payment_data["currency"]
            )
        except httpx.HTTPStatusError as e:
            # 處理 API 錯誤
            error_data = e.response.json()
            raise HTTPException(
                status_code=400,
                detail=f"Payment failed: {error_data.get('error', {}).get('message', 'Unknown error')}"
            )
        except httpx.RequestError as e:
            # 處理連接錯誤
            raise HTTPException(
                status_code=503,
                detail=f"Service unavailable: {str(e)}"
            )

# 客戶端和服務依賴
def get_payment_client():
    return httpx.Client(
        base_url=settings.payment_api_url,
        headers={"Authorization": f"Bearer {settings.payment_api_key}"}
    )

def get_payment_service(client: httpx.Client = Depends(get_payment_client)):
    return PaymentService(client)

@app.post("/payments", response_model=PaymentResponse)
def create_payment(
    payment: PaymentRequest,
    payment_service: PaymentService = Depends(get_payment_service)
):
    return payment_service.create_payment(payment)
```

**優點**：
- 更好的關注點分離
- 增強可測試性
- 更好的錯誤處理和類型安全

**缺點**：
- 代碼量增加
- 需要維護更多的類和接口

### 2. 異步服務整合

對於 I/O 密集型的外部服務調用，使用異步客戶端可以提高性能。

```python
from fastapi import FastAPI, Depends, HTTPException
import httpx
import asyncio
from pydantic import BaseSettings
from typing import List, Dict, Any

class Settings(BaseSettings):
    search_api_key: str
    search_api_url: str = "https://api.algolia.com/1"
    
    class Config:
        env_file = ".env"

settings = Settings()
app = FastAPI()

# 全局異步客戶端
search_client = None

@app.on_event("startup")
async def startup_event():
    global search_client
    search_client = httpx.AsyncClient(
        base_url=settings.search_api_url,
        headers={
            "X-Algolia-API-Key": settings.search_api_key,
            "X-Algolia-Application-Id": "YOUR_APP_ID"
        }
    )

@app.on_event("shutdown")
async def shutdown_event():
    global search_client
    if search_client:
        await search_client.aclose()

async def get_search_client():
    return search_client

class SearchService:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
    
    async def search(self, index: str, query: str) -> Dict[str, Any]:
        response = await self.client.post(
            f"/indexes/{index}/query",
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()
    
    async def multi_search(self, indexes: List[str], query: str) -> List[Dict[str, Any]]:
        # 並行執行多個搜索
        tasks = [self.search(index, query) for index in indexes]
        results = await asyncio.gather(*tasks)
        
        return [
            {"index": index, "results": result}
            for index, result in zip(indexes, results)
        ]

def get_search_service(client: httpx.AsyncClient = Depends(get_search_client)):
    return SearchService(client)

@app.get("/search")
async def search(
    query: str,
    indexes: str,  # 逗號分隔的索引列表
    search_service: SearchService = Depends(get_search_service)
):
    index_list = [index.strip() for index in indexes.split(",")]
    
    try:
        results = await search_service.multi_search(index_list, query)
        return {"query": query, "results": results}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Search service unavailable: {str(e)}")
```

**優點**：
- 支持並行請求
- 提高 I/O 密集型操作的性能
- 更好地利用系統資源

**缺點**：
- 異步代碼可能更難理解和調試
- 需要處理異步上下文中的錯誤

### 3. 重試機制與斷路器模式

對於不可靠的外部服務，實現重試機制和斷路器模式可以提高系統的穩定性。

```python
from fastapi import FastAPI, Depends, HTTPException
import httpx
import time
import random
from functools import wraps
from pydantic import BaseSettings
from typing import Callable, TypeVar, Any

# 定義泛型類型
T = TypeVar("T")

class Settings(BaseSettings):
    notification_api_key: str
    notification_api_url: str = "https://api.pushover.net/1"
    
    class Config:
        env_file = ".env"

settings = Settings()
app = FastAPI()

# 重試裝飾器
def retry(max_retries: int = 3, backoff_factor: float = 0.5):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    retries += 1
                    if retries > max_retries:
                        raise
                    
                    # 指數退避
                    wait_time = backoff_factor * (2 ** (retries - 1)) * (1 + random.random())
                    print(f"Retrying after {wait_time:.2f}s due to {str(e)}")
                    await asyncio.sleep(wait_time)
        
        return wrapper
    return decorator

# 斷路器狀態
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_time: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
    
    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"
    
    def allow_request(self) -> bool:
        if self.state == "CLOSED":
            return True
        
        # 檢查是否應該嘗試恢復
        if self.state == "OPEN" and time.time() - self.last_failure_time > self.recovery_time:
            self.state = "HALF-OPEN"
        
        return self.state == "HALF-OPEN"

# 通知服務
notification_circuit_breaker = CircuitBreaker()

class NotificationService:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
    
    @retry(max_retries=3)
    async def send_notification(self, user_key: str, message: str, title: str = None) -> bool:
        if not notification_circuit_breaker.allow_request():
            raise HTTPException(
                status_code=503,
                detail="Notification service is currently unavailable"
            )
        
        try:
            response = await self.client.post(
                "/messages.json",
                data={
                    "token": settings.notification_api_key,
                    "user": user_key,
                    "message": message,
                    "title": title
                }
            )
            response.raise_for_status()
            
            notification_circuit_breaker.record_success()
            return True
        except Exception as e:
            notification_circuit_breaker.record_failure()
            raise

# 依賴函數
async def get_notification_client():
    return httpx.AsyncClient(base_url=settings.notification_api_url)

def get_notification_service(client: httpx.AsyncClient = Depends(get_notification_client)):
    return NotificationService(client)

@app.post("/notify")
async def send_notification(
    user_key: str,
    message: str,
    title: str = None,
    notification_service: NotificationService = Depends(get_notification_service)
):
    try:
        await notification_service.send_notification(user_key, message, title)
        return {"status": "Notification sent successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send notification: {str(e)}"
        )
```

**優點**：
- 提高系統穩定性
- 防止外部服務故障導致系統崩潰
- 提供優雅的降級機制

**缺點**：
- 增加代碼複雜性
- 需要謹慎配置重試參數

## 測試外部服務依賴

測試與外部服務的集成是一個挑戰，但 FastAPI 的依賴注入系統使其變得更容易。

### 1. 使用模擬對象

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from your_app import app, PaymentService

@pytest.fixture
def mock_payment_service():
    # 創建模擬服務
    mock_service = MagicMock(spec=PaymentService)
    
    # 配置模擬方法
    mock_service.create_payment.return_value = {
        "payment_id": "test_payment_123",
        "status": "succeeded",
        "amount": 1000,
        "currency": "usd"
    }
    
    return mock_service

@pytest.fixture
def client(mock_payment_service):
    # 覆蓋依賴
    app.dependency_overrides[get_payment_service] = lambda: mock_payment_service
    
    with TestClient(app) as client:
        yield client
    
    # 清理
    app.dependency_overrides = {}

def test_create_payment(client, mock_payment_service):
    response = client.post(
        "/payments",
        json={
            "amount": 1000,
            "currency": "usd",
            "description": "Test payment"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["payment_id"] == "test_payment_123"
    assert data["status"] == "succeeded"
    
    # 驗證模擬服務被正確調用
    mock_payment_service.create_payment.assert_called_once()
    call_args = mock_payment_service.create_payment.call_args[0][0]
    assert call_args.amount == 1000
    assert call_args.currency == "usd"
```

### 2. 使用 httpx 的響應模擬

對於直接使用 httpx 客戶端的依賴，我們可以使用 `httpx.MockResponse`。

```python
import pytest
from fastapi.testclient import TestClient
import httpx
import json
from unittest.mock import patch
from your_app import app

# 模擬 httpx 響應
@pytest.fixture
def mock_weather_response():
    return httpx.Response(
        status_code=200,
        json={
            "location": {"name": "London", "country": "UK"},
            "current": {
                "temp_c": 15.0,
                "condition": {"text": "Partly cloudy"}
            }
        }
    )

# 測試使用模擬響應
@patch("httpx.Client.get")
def test_get_weather(mock_get, mock_weather_response):
    mock_get.return_value = mock_weather_response
    
    with TestClient(app) as client:
        response = client.get("/weather/London")
    
    assert response.status_code == 200
    data = response.json()
    assert data["city"] == "London"
    assert data["temperature"] == 15.0
    assert data["condition"] == "Partly cloudy"
    
    # 驗證 httpx 調用
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == "/current.json"
    assert kwargs["params"]["q"] == "London"
```

### 3. 使用測試服務器

對於更複雜的集成測試，我們可以設置一個測試服務器來模擬外部 API。

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import uvicorn
import threading
import time
import requests
from your_app import app as main_app

# 創建模擬 API 服務器
mock_api = FastAPI()

@mock_api.get("/api/products")
def mock_get_products():
    return [
        {"id": 1, "name": "Product 1", "price": 10.99},
        {"id": 2, "name": "Product 2", "price": 20.99}
    ]

@mock_api.get("/api/products/{product_id}")
def mock_get_product(product_id: int):
    products = {
        1: {"id": 1, "name": "Product 1", "price": 10.99},
        2: {"id": 2, "name": "Product 2", "price": 20.99}
    }
    
    if product_id not in products:
        return {"error": "Product not found"}, 404
    
    return products[product_id]

# 啟動模擬服務器
@pytest.fixture(scope="module")
def mock_api_server():
    # 在單獨的線程中啟動服務器
    thread = threading.Thread(
        target=uvicorn.run,
        args=(mock_api,),
        kwargs={"host": "127.0.0.1", "port": 8001, "log_level": "error"},
        daemon=True
    )
    thread.start()
    
    # 等待服務器啟動
    time.sleep(1)
    
    yield "http://127.0.0.1:8001"
    
    # 不需要顯式停止，因為使用了 daemon=True

# 配置主應用使用模擬服務器
@pytest.fixture
def client(mock_api_server):
    # 覆蓋配置
    main_app.dependency_overrides[get_product_api_url] = lambda: mock_api_server
    
    with TestClient(main_app) as client:
        yield client
    
    main_app.dependency_overrides = {}

# 測試與模擬 API 的集成
def test_get_product_details(client):
    response = client.get("/products/1")
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert data["name"] == "Product 1"
    assert data["price"] == 10.99
```

## 高級技巧與最佳實踐

### 1. 使用工廠模式創建服務客戶端

工廠模式可以幫助我們根據不同的配置或環境創建適當的服務客戶端。

```python
from fastapi import FastAPI, Depends
import httpx
from pydantic import BaseSettings
from enum import Enum
from typing import Optional

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Settings(BaseSettings):
    environment: Environment = Environment.DEVELOPMENT
    payment_api_key: str
    payment_api_url_dev: str = "https://sandbox.payment.com/v1"
    payment_api_url_staging: str = "https://staging.payment.com/v1"
    payment_api_url_prod: str = "https://api.payment.com/v1"
    
    class Config:
        env_file = ".env"

settings = Settings()
app = FastAPI()

# 客戶端工廠
class PaymentClientFactory:
    @staticmethod
    def create_client(environment: Environment, api_key: str) -> httpx.Client:
        base_urls = {
            Environment.DEVELOPMENT: settings.payment_api_url_dev,
            Environment.STAGING: settings.payment_api_url_staging,
            Environment.PRODUCTION: settings.payment_api_url_prod
        }
        
        base_url = base_urls.get(environment)
        
        return httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )

# 依賴函數
def get_payment_client(env: Optional[Environment] = None):
    # 使用提供的環境或默認環境
    environment = env or settings.environment
    
    return PaymentClientFactory.create_client(
        environment=environment,
        api_key=settings.payment_api_key
    )

@app.get("/payment-status/{payment_id}")
def get_payment_status(
    payment_id: str, 
    client: httpx.Client = Depends(get_payment_client)
):
    response = client.get(f"/payments/{payment_id}")
    response.raise_for_status()
    
    return response.json()

# 可以為特定路由指定環境
@app.post("/test-payment")
def create_test_payment(
    amount: int,
    client: httpx.Client = Depends(lambda: get_payment_client(Environment.DEVELOPMENT))
):
    # 總是使用開發環境進行測試支付
    response = client.post("/payments", json={"amount": amount})
    response.raise_for_status()
    
    return response.json()
```

### 2. 使用緩存減少外部調用

對於頻繁訪問但不常變化的外部數據，使用緩存可以減少外部服務調用。

```python
from fastapi import FastAPI, Depends
import httpx
from functools import lru_cache
import time
from datetime import datetime, timedelta

app = FastAPI()

# 緩存結果的類
class CachedResponse:
    def __init__(self, data, expires_at):
        self.data = data
        self.expires_at = expires_at
    
    @property
    def is_expired(self):
        return datetime.now() > self.expires_at

# 緩存裝飾器
def cache_response(ttl_seconds=300):
    def decorator(func):
        cache = {}
        
        async def wrapper(*args, **kwargs):
            # 創建緩存鍵
            cache_key = str(args) + str(kwargs)
            
            # 檢查緩存
            if cache_key in cache and not cache[cache_key].is_expired:
                print(f"Cache hit for {cache_key}")
                return cache[cache_key].data
            
            # 調用原始函數
            result = await func(*args, **kwargs)
            
            # 存儲結果到緩存
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            cache[cache_key] = CachedResponse(result, expires_at)
            
            return result
        
        return wrapper
    
    return decorator

# 外部服務客戶端
class ExchangeRateService:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
    
    @cache_response(ttl_seconds=3600)  # 匯率每小時緩存一次
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        response = await self.client.get(
            "/latest",
            params={"base": from_currency, "symbols": to_currency}
        )
        response.raise_for_status()
        
        data = response.json()
        return data["rates"][to_currency]

# 依賴函數
async def get_exchange_client():
    return httpx.AsyncClient(base_url="https://api.exchangerate.host")

def get_exchange_service(client: httpx.AsyncClient = Depends(get_exchange_client)):
    return ExchangeRateService(client)

@app.get("/convert")
async def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    exchange_service: ExchangeRateService = Depends(get_exchange_service)
):
    rate = await exchange_service.get_exchange_rate(from_currency, to_currency)
    converted_amount = amount * rate
    
    return {
        "original": {"amount": amount, "currency": from_currency},
        "converted": {"amount": converted_amount, "currency": to_currency},
        "rate": rate
    }
```

### 3. 超時和限流管理

對外部服務的請求應該設置適當的超時，並實施限流以避免過度使用外部資源。

```python
from fastapi import FastAPI, Depends, HTTPException
import httpx
import asyncio
import time
from pydantic import BaseSettings

class Settings(BaseSettings):
    api_key: str
    api_url: str = "https://api.example.com/v1"
    
    class Config:
        env_file = ".env"

settings = Settings()
app = FastAPI()

# 限流器
class RateLimiter:
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
    
    async def wait(self):
        # 計算需要等待的時間
        now = time.time()
        elapsed = now - self.last_call_time
        
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            await asyncio.sleep(wait_time)
        
        self.last_call_time = time.time()

# 創建服務類
class ApiService:
    def __init__(self, client: httpx.AsyncClient, rate_limiter: RateLimiter):
        self.client = client
        self.rate_limiter = rate_limiter
    
    async def make_request(self, endpoint: str, **kwargs):
        # 應用限流
        await self.rate_limiter.wait()
        
        try:
            # 設置超時
            timeout = httpx.Timeout(10.0, connect=5.0)
            response = await self.client.get(endpoint, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="Request to external service timed out"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"External service error: {str(e)}"
            )

# 依賴函數
async def get_api_client():
    return httpx.AsyncClient(
        base_url=settings.api_url,
        headers={"Authorization": f"Bearer {settings.api_key}"}
    )

def get_rate_limiter():
    # 限制為每秒 5 個請求
    return RateLimiter(calls_per_second=5.0)

def get_api_service(
    client: httpx.AsyncClient = Depends(get_api_client),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    return ApiService(client, rate_limiter)

@app.get("/api-proxy/{endpoint}")
async def api_proxy(
    endpoint: str,
    api_service: ApiService = Depends(get_api_service)
):
    return await api_service.make_request(f"/{endpoint}")
```

## 最佳實踐

1. **使用適當的抽象**：通過服務類或接口封裝外部服務的交互，使代碼更易於測試和維護。

2. **妥善管理客戶端生命週期**：根據服務特性選擇適當的客戶端生命週期管理策略。

3. **實施超時和重試機制**：為所有外部調用設置合理的超時，並在適當的情況下實施重試。

4. **使用斷路器模式**：防止外部服務故障影響整個系統。

5. **實施限流**：避免超過外部服務的使用限制。

6. **緩存不常變化的數據**：減少對外部服務的請求次數。

7. **使用非同步客戶端**：對於 I/O 密集型操作，使用異步客戶端提高性能。

8. **妥善處理錯誤**：捕獲和處理外部服務可能返回的各種錯誤。

```python
async def call_external_service(client, endpoint):
    try:
        response = await client.get(endpoint, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        # 處理超時
        logger.warning(f"Request to {endpoint} timed out")
        raise HTTPException(status_code=504, detail="Service timeout")
    except httpx.HTTPStatusError as e:
        # 處理 HTTP 錯誤
        logger.error(f"HTTP error from {endpoint}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code >= 500:
            raise HTTPException(status_code=502, detail="External service error")
        else:
            raise HTTPException(status_code=400, detail="Invalid request to external service")
    except httpx.RequestError as e:
        # 處理連接錯誤
        logger.error(f"Connection error to {endpoint}: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")
```

9. **使用環境變數進行配置**：不要在代碼中硬編碼 API 密鑰和 URL。

10. **編寫全面的測試**：使用模擬對象或測試服務器測試與外部服務的集成。

## 結論

FastAPI 的依賴注入系統為外部服務的整合提供了強大的支持。通過適當的抽象、生命週期管理和錯誤處理，我們可以構建出既靈活又穩定的外部服務集成。

從簡單的 API 客戶端到複雜的服務類，從基本的錯誤處理到高級的重試和斷路器模式，依賴注入使得外部服務的整合變得更加結構化和可維護。

通過遵循本文介紹的最佳實踐和模式，您可以確保您的 FastAPI 應用能夠可靠地與外部世界交互，同時保持代碼的可測試性和可維護性。

記住，與外部服務的整合總是存在風險，因為這些服務不在您的控制之下。通過實施適當的錯誤處理、超時、重試和降級策略，您可以構建出能夠優雅地處理這些風險的應用。