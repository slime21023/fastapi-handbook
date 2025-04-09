# Mocking 技術

## Mocking 的基本概念

Mocking 是一種測試技術，通過創建模擬對象來替代真實的依賴項，使測試能夠在隔離的環境中進行。在 FastAPI 應用程序的測試中，Mocking 技術尤為重要，因為它可以幫助我們測試複雜的系統，而無需依賴外部服務或資源。

| 概念 | 說明 |
|------|------|
| **Mock 對象** | 模擬真實對象行為的假對象，可以預設返回值和記錄調用 |
| **Stub** | 提供預定義響應的簡單實現，不記錄調用信息 |
| **Spy** | 記錄調用信息但保留原始行為的對象 |
| **Fake** | 具有實際工作實現但不適合生產的簡化版本 |
| **Dummy** | 傳遞但不實際使用的對象 |

## Python 中的 Mocking 工具

Python 的標準庫提供了強大的 Mocking 工具，主要通過 `unittest.mock` 模塊實現。

### 主要的 Mock 類型

```python
from unittest.mock import Mock, MagicMock, AsyncMock, patch, PropertyMock
```

| Mock 類型 | 用途 |
|----------|------|
| **Mock** | 基本的模擬對象，可以設置返回值和副作用 |
| **MagicMock** | 增強版的 Mock，預先實現了許多魔術方法 |
| **AsyncMock** | 用於模擬非同步函數和方法的特殊 Mock |
| **PropertyMock** | 用於模擬屬性的特殊 Mock |
| **patch** | 用於臨時替換模塊和類的裝飾器和上下文管理器 |

## 基本的 Mocking 技術

### 創建和配置 Mock 對象

```python
# tests/test_basic_mocking.py
from unittest.mock import Mock

def test_basic_mock():
    # 創建一個基本的 Mock 對象
    mock_object = Mock()
    
    # 配置返回值
    mock_object.method.return_value = "mocked result"
    
    # 調用方法
    result = mock_object.method()
    
    # 驗證結果
    assert result == "mocked result"
    
    # 驗證調用
    mock_object.method.assert_called_once()
```

### 使用 patch 裝飾器

```python
# tests/test_patch_decorator.py
from unittest.mock import patch
import requests
from app.services.user_service import get_user_data

@patch("app.services.user_service.requests.get")
def test_get_user_data(mock_get):
    # 配置模擬響應
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": 1, "name": "Test User"}
    mock_get.return_value = mock_response
    
    # 調用被測試的函數
    user = get_user_data(1)
    
    # 驗證結果
    assert user["name"] == "Test User"
    
    # 驗證 requests.get 被正確調用
    mock_get.assert_called_once_with("https://api.example.com/users/1")
```

### 使用 patch 上下文管理器

```python
# tests/test_patch_context_manager.py
from unittest.mock import patch, Mock
from app.services.email_service import send_welcome_email

def test_send_welcome_email():
    # 創建一個用戶對象
    user = {"email": "user@example.com", "name": "Test User"}
    
    # 使用上下文管理器模擬 smtp 庫
    with patch("app.services.email_service.smtplib.SMTP") as mock_smtp:
        # 配置模擬對象
        mock_smtp_instance = Mock()
        mock_smtp.return_value = mock_smtp_instance
        
        # 調用被測試的函數
        result = send_welcome_email(user)
        
        # 驗證結果
        assert result is True
        
        # 驗證 SMTP 方法被正確調用
        mock_smtp.assert_called_once_with("smtp.example.com", 587)
        mock_smtp_instance.send_message.assert_called_once()
```

## 模擬 FastAPI 依賴項

FastAPI 的依賴注入系統是其核心特性之一，測試時需要有效地模擬這些依賴項。

### 使用 app.dependency_overrides

FastAPI 提供了一種覆蓋依賴項的機制，這在測試中非常有用：

```python
# tests/api/test_dependency_override.py
from fastapi.testclient import TestClient
from app.main import app
from app.dependencies.auth import get_current_user

client = TestClient(app)

def test_protected_endpoint_with_override():
    # 創建一個替代依賴
    def override_get_current_user():
        return {"id": 1, "username": "testuser"}
    
    # 覆蓋依賴
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    try:
        # 發送請求
        response = client.get("/items/me")
        
        # 驗證響應
        assert response.status_code == 200
        assert response.json() == {"owner_id": 1, "items": []}
    finally:
        # 清理依賴覆蓋
        app.dependency_overrides = {}
```

## 模擬數據庫操作

測試數據庫操作時，我們通常希望避免實際的數據庫交互，可以通過模擬 ORM 層或使用內存數據庫來實現。

### 模擬 SQLAlchemy 會話

```python
# tests/repositories/test_user_repository.py
from unittest.mock import Mock
from app.repositories.user_repository import UserRepository
from app.models.user import User

def test_get_user_by_id():
    # 創建模擬的 SQLAlchemy 會話
    mock_session = Mock()
    
    # 配置模擬查詢結果
    mock_user = Mock(spec=User)
    mock_user.id = 1
    mock_user.username = "testuser"
    mock_user.email = "test@example.com"
    
    # 配置模擬查詢鏈
    mock_session.query.return_value.filter.return_value.first.return_value = mock_user
    
    # 創建存儲庫並注入模擬會話
    repo = UserRepository(mock_session)
    
    # 調用方法
    user = repo.get_by_id(1)
    
    # 驗證結果
    assert user.id == 1
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    
    # 驗證查詢鏈
    mock_session.query.assert_called_once_with(User)
```

### 模擬非同步 SQLAlchemy 會話

```python
# tests/repositories/test_async_user_repository.py
import pytest
from unittest.mock import AsyncMock, Mock
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.async_user_repository import AsyncUserRepository
from app.models.user import User

@pytest.mark.asyncio
async def test_async_get_user_by_id():
    # 創建模擬的非同步 SQLAlchemy 會話
    mock_session = AsyncMock(spec=AsyncSession)
    
    # 配置模擬查詢結果
    mock_user = Mock(spec=User)
    mock_user.id = 1
    mock_user.username = "testuser"
    mock_user.email = "test@example.com"
    
    # 配置模擬執行結果
    mock_result = Mock()
    mock_result.scalars.return_value.first.return_value = mock_user
    mock_session.execute.return_value = mock_result
    
    # 創建存儲庫並注入模擬會話
    repo = AsyncUserRepository(mock_session)
    
    # 調用方法
    user = await repo.get_by_id(1)
    
    # 驗證結果
    assert user.id == 1
    assert user.username == "testuser"
    assert user.email == "test@example.com"
```

## 模擬外部 API 和服務

在測試中，我們通常需要模擬外部 API 和服務的響應，以避免實際的網絡請求。

### 模擬 HTTP 請求

```python
# tests/services/test_external_api_service.py
from unittest.mock import patch, Mock
from app.services.weather_service import get_current_weather

def test_get_current_weather():
    # 模擬成功的 API 響應
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "location": "New York",
        "temperature": 22,
        "conditions": "Sunny"
    }
    
    # 使用 patch 模擬 requests.get
    with patch("app.services.weather_service.requests.get", return_value=mock_response) as mock_get:
        # 調用服務
        weather = get_current_weather("New York")
        
        # 驗證結果
        assert weather["temperature"] == 22
        assert weather["conditions"] == "Sunny"
        
        # 驗證 API 調用
        mock_get.assert_called_once_with(
            "https://api.weatherservice.com/current",
            params={"city": "New York", "units": "metric"}
        )
```

### 模擬非同步 HTTP 請求

```python
# tests/services/test_async_external_api_service.py
import pytest
from unittest.mock import patch, AsyncMock
from app.services.async_weather_service import get_current_weather_async

@pytest.mark.asyncio
async def test_get_current_weather_async():
    # 創建模擬的非同步響應
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "location": "New York",
        "temperature": 22,
        "conditions": "Sunny"
    }
    
    # 模擬 httpx.AsyncClient.get 方法
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        
        # 調用非同步服務
        weather = await get_current_weather_async("New York")
        
        # 驗證結果
        assert weather["temperature"] == 22
        assert weather["conditions"] == "Sunny"
```

## 模擬文件系統操作

測試涉及文件系統操作的代碼時，我們可以模擬文件系統來避免實際的文件操作。

### 使用 pyfakefs 模擬文件系統

```bash
pip install pyfakefs
```

```python
# tests/utils/test_file_utils.py
import pytest
from pyfakefs.fake_filesystem_unittest import Patcher
from app.utils.file_utils import save_user_avatar

def test_save_user_avatar():
    # 創建模擬的文件數據
    avatar_data = b"fake image data"
    user_id = 123
    
    # 使用 pyfakefs 模擬文件系統
    with Patcher() as patcher:
        # 創建必要的目錄結構
        patcher.fs.create_dir("/app/uploads/avatars")
        
        # 調用被測試的函數
        file_path = save_user_avatar(user_id, avatar_data)
        
        # 驗證文件路徑
        assert file_path == f"/app/uploads/avatars/{user_id}.png"
        
        # 驗證文件內容
        with open(file_path, "rb") as f:
            saved_data = f.read()
            assert saved_data == avatar_data
```

## 模擬時間和日期

測試涉及時間和日期的代碼時，我們需要能夠控制當前時間，以便測試各種情況。

### 使用 freezegun 模擬時間

```bash
pip install freezegun
```

```python
# tests/utils/test_time_utils.py
from datetime import datetime, timedelta
from freezegun import freeze_time
from app.utils.time_utils import is_token_expired, get_token_expiration

def test_is_token_expired():
    # 創建一個過期的令牌時間（1小時前）
    expired_time = datetime.utcnow() - timedelta(hours=1)
    
    # 測試過期的令牌
    assert is_token_expired(expired_time) is True
    
    # 創建一個未過期的令牌時間（1小時後）
    valid_time = datetime.utcnow() + timedelta(hours=1)
    
    # 測試未過期的令牌
    assert is_token_expired(valid_time) is False

@freeze_time("2023-01-01 12:00:00")
def test_get_token_expiration():
    # 在凍結的時間點調用函數
    expiration = get_token_expiration()
    
    # 驗證結果 - 應該是凍結時間加上 24 小時
    expected = datetime(2023, 1, 2, 12, 0, 0)
    assert expiration == expected
```

## 模擬環境變量

測試依賴於環境變量的代碼時，我們需要能夠控制這些變量的值。

### 使用 monkeypatch 模擬環境變量

```python
# tests/config/test_settings.py
from app.config import get_settings

def test_get_settings_development(monkeypatch):
    # 設置環境變量
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DATABASE_URL", "postgresql://dev:dev@localhost/dev_db")
    monkeypatch.setenv("API_KEY", "dev_api_key")
    
    # 獲取設置
    settings = get_settings()
    
    # 驗證設置
    assert settings.environment == "development"
    assert settings.database_url == "postgresql://dev:dev@localhost/dev_db"
    assert settings.api_key == "dev_api_key"
    assert settings.debug is True  # 開發環境默認啟用調試
```

## 高級 Mocking 技術

### 模擬異常和副作用

```python
# tests/services/test_exception_handling.py
from unittest.mock import patch
import pytest
import requests
from app.services.user_service import get_user_data

def test_get_user_data_connection_error():
    # 模擬 requests.get 拋出連接錯誤
    with patch("app.services.user_service.requests.get", side_effect=requests.ConnectionError("Connection failed")):
        # 調用函數並驗證它處理異常
        with pytest.raises(ValueError) as exc_info:
            get_user_data(1)
        
        # 驗證錯誤消息
        assert "Failed to connect to API" in str(exc_info.value)
```

### 使用 side_effect 模擬複雜行為

```python
# tests/services/test_side_effects.py
from unittest.mock import Mock, patch

def test_retry_mechanism():
    # 創建一個計數器來跟踪調用次數
    call_count = 0
    
    # 定義一個 side_effect 函數，前兩次拋出異常，第三次返回成功
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Connection failed")
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "success"}
        return mock_response
    
    # 模擬 requests.get 使用我們的 side_effect
    with patch("app.services.resilient_service.requests.get", side_effect=side_effect):
        from app.services.resilient_service import fetch_with_retry
        
        # 調用具有重試機制的函數
        result = fetch_with_retry("https://api.example.com/data")
        
        # 驗證結果
        assert result == {"data": "success"}
        assert call_count == 3  # 驗證函數被調用了 3 次
```

## 模擬 FastAPI 請求和響應對象

測試 FastAPI 的依賴項和中間件時，我們需要模擬 FastAPI 的請求和響應對象。

### 模擬 FastAPI 請求

```python
# tests/middleware/test_request_logging.py
from unittest.mock import Mock, patch
from fastapi import Request
from app.middleware.request_logging import log_request

async def test_log_request_middleware():
    # 創建模擬的請求對象
    mock_request = Mock(spec=Request)
    mock_request.method = "GET"
    mock_request.url.path = "/api/users"
    mock_request.headers = {"User-Agent": "Test Client"}
    mock_request.client.host = "127.0.0.1"
    
    # 創建模擬的 call_next 函數
    async def mock_call_next(request):
        return Mock(status_code=200)
    
    # 模擬日誌記錄器
    with patch("app.middleware.request_logging.logger") as mock_logger:
        # 調用中間件
        response = await log_request(mock_request, mock_call_next)
        
        # 驗證響應
        assert response.status_code == 200
        
        # 驗證日誌記錄
        mock_logger.info.assert_called_once()
```

## 模擬 WebSocket 連接

測試 WebSocket 端點時，我們需要模擬 WebSocket 連接。

```python
# tests/api/test_websocket.py
from unittest.mock import AsyncMock, patch
import pytest
from fastapi import WebSocket, WebSocketDisconnect
from app.routers.websocket import websocket_endpoint

@pytest.mark.asyncio
async def test_websocket_endpoint():
    # 創建模擬的 WebSocket 對象
    mock_websocket = AsyncMock(spec=WebSocket)
    
    # 配置 receive_text 方法，第一次返回消息，第二次拋出異常
    mock_websocket.receive_text.side_effect = [
        "Hello WebSocket",
        WebSocketDisconnect()
    ]
    
    # 調用 WebSocket 端點
    with patch("app.routers.websocket.manager") as mock_manager:
        try:
            await websocket_endpoint(mock_websocket, 1)
        except WebSocketDisconnect:
            pass
        
        # 驗證 WebSocket 方法被調用
        mock_websocket.accept.assert_called_once()
        mock_websocket.receive_text.assert_called()
        
        # 驗證連接管理器方法被調用
        mock_manager.connect.assert_called_once_with(mock_websocket)
        mock_manager.disconnect.assert_called_once_with(mock_websocket)
```

## 模擬 Redis 和緩存操作

測試使用 Redis 或其他緩存系統的代碼時，我們可以模擬這些依賴項。

```python
# tests/services/test_cache_service.py
from unittest.mock import Mock, patch
from app.services.cache_service import get_cached_user

def test_get_cached_user_hit():
    # 模擬 Redis 客戶端
    mock_redis = Mock()
    mock_redis.get.return_value = '{"id": 1, "username": "cacheduser"}'
    
    # 使用 patch 模擬 Redis 連接
    with patch("app.services.cache_service.redis_client", mock_redis):
        # 調用緩存服務
        user = get_cached_user(1)
        
        # 驗證結果
        assert user["id"] == 1
        assert user["username"] == "cacheduser"
        
        # 驗證 Redis 調用
        mock_redis.get.assert_called_once_with("user:1")
```

## 模擬 JWT 令牌驗證

測試涉及 JWT 令牌驗證的代碼時，我們需要模擬令牌解碼和驗證過程。

```python
# tests/auth/test_jwt.py
from unittest.mock import patch
import pytest
from app.auth.jwt import verify_token

def test_verify_token_valid():
    # 模擬 jwt.decode 返回有效的負載
    with patch("app.auth.jwt.jwt.decode") as mock_decode:
        mock_decode.return_value = {"sub": "user123", "exp": 9999999999}
        
        # 驗證令牌
        user_id = verify_token("valid_token")
        
        # 驗證結果
        assert user_id == "user123"
        mock_decode.assert_called_once()

def test_verify_token_expired():
    # 模擬 jwt.decode 拋出過期異常
    with patch("app.auth.jwt.jwt.decode") as mock_decode:
        from jwt.exceptions import ExpiredSignatureError
        mock_decode.side_effect = ExpiredSignatureError("Token expired")
        
        # 驗證令牌
        user_id = verify_token("expired_token")
        
        # 驗證結果
        assert user_id is None
        mock_decode.assert_called_once()
```

## 模擬 OAuth2 認證流程

測試 OAuth2 認證流程時，我們需要模擬授權服務器的響應。

```python
# tests/auth/test_oauth2.py
from unittest.mock import patch, Mock
import pytest
from app.auth.oauth2 import get_token_from_code

def test_get_token_from_code_success():
    # 模擬成功的令牌響應
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "mock_access_token",
        "token_type": "bearer",
        "expires_in": 3600
    }
    
    # 模擬 requests.post
    with patch("app.auth.oauth2.requests.post", return_value=mock_response) as mock_post:
        # 獲取令牌
        token_info = get_token_from_code("auth_code")
        
        # 驗證結果
        assert token_info["access_token"] == "mock_access_token"
        assert token_info["token_type"] == "bearer"
        
        # 驗證 API 調用
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert "code=auth_code" in call_args["data"]
```

## 模擬數據庫遷移和種子數據

測試涉及數據庫遷移和種子數據的代碼時，我們可以模擬這些操作。

```python
# tests/db/test_migrations.py
from unittest.mock import patch, Mock
import pytest
from app.db.migrations import run_migrations

def test_run_migrations():
    # 模擬 Alembic 命令
    with patch("app.db.migrations.command") as mock_command:
        # 調用遷移函數
        run_migrations()
        
        # 驗證 Alembic 命令被調用
        mock_command.upgrade.assert_called_once_with("head")
```

## Mocking 的最佳實踐

### 只模擬直接依賴項

模擬應該集中在被測試代碼直接依賴的組件上，而不是更深層次的依賴項。這樣可以確保測試更加集中和可維護。

```python
# 好的做法
def test_user_service():
    # 直接模擬用戶存儲庫
    mock_repo = Mock()
    mock_repo.get_by_id.return_value = {"id": 1, "name": "Test User"}
    
    # 注入模擬依賴
    service = UserService(user_repository=mock_repo)
    
    # 測試服務
    user = service.get_user(1)
    assert user["name"] == "Test User"
```

### 避免過度模擬

過度模擬會使測試變得脆弱，難以維護，並且可能無法發現實際問題。

```python
# 避免這樣做
def test_over_mocking():
    # 模擬太多細節
    with patch("module.Class") as MockClass:
        instance = MockClass.return_value
        instance.method1.return_value.method2.return_value.method3.return_value = "result"
        
        # 這樣的測試與實現細節過度耦合
```

### 使用夾具（Fixtures）組織模擬

使用 pytest 夾具來組織和重用模擬對象，使測試更加清晰和可維護。

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_user_repo():
    mock_repo = Mock()
    mock_repo.get_by_id.return_value = {"id": 1, "name": "Test User"}
    mock_repo.get_by_email.return_value = {"id": 1, "email": "test@example.com"}
    return mock_repo

@pytest.fixture
def mock_auth_service():
    with patch("app.services.auth_service.AuthService") as MockAuthService:
        instance = MockAuthService.return_value
        instance.verify_password.return_value = True
        instance.create_token.return_value = "mock_token"
        yield instance
```

### 驗證模擬調用

不僅要測試結果，還要驗證模擬對象是否按預期被調用。

```python
def test_create_user(mock_user_repo, mock_email_service):
    # 創建服務
    service = UserService(user_repository=mock_user_repo, email_service=mock_email_service)
    
    # 調用方法
    service.create_user({"name": "New User", "email": "new@example.com"})
    
    # 驗證存儲庫方法被調用
    mock_user_repo.create.assert_called_once()
    
    # 驗證電子郵件服務被調用
    mock_email_service.send_welcome_email.assert_called_once_with("new@example.com")
```

## 總結

Mocking 是測試 FastAPI 應用程序的強大工具，它允許我們在隔離的環境中測試代碼，而無需依賴外部服務或資源。通過使用 Python 的 `unittest.mock` 模塊和其他工具，我們可以有效地模擬各種依賴項，包括數據庫、外部 API、文件系統和時間。

關鍵點：

1. **選擇正確的模擬類型**：根據需要選擇 Mock、MagicMock、AsyncMock 或其他類型。

2. **只模擬直接依賴項**：避免過度模擬，專注於被測試代碼的直接依賴項。

3. **使用 patch 裝飾器和上下文管理器**：臨時替換模塊和類，使測試更加隔離。

4. **驗證模擬調用**：確保模擬對象按預期被調用，包括調用次數、參數等。

5. **使用夾具組織模擬**：通過 pytest 夾具重用模擬對象，使測試更加清晰和可維護。

6. **模擬非同步代碼**：使用 AsyncMock 模擬非同步函數和方法，確保非同步代碼的正確測試。

通過掌握這些 Mocking 技術，你可以編寫更加健壯、可靠的測試，確保 FastAPI 應用程序的質量和穩定性。